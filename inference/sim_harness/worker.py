from __future__ import annotations

from dataclasses import asdict
from time import time
import asyncio

from .agent_rpc import request_frame, request_status, request_stop, send_input
from .config import HarnessConfig, SlotConfig
from .models import DiscoveredEndpoint, SlotSnapshot, SlotStatus
from .process_control import launch_slot_session, tmux_kill_session


class SlotWorker:
    def __init__(self, harness: HarnessConfig, slot: SlotConfig) -> None:
        self.harness = harness
        self.slot = slot
        self.endpoint: DiscoveredEndpoint | None = None
        self.capture: SlotCaptureBridge | None = None
        self.status = SlotStatus.STOPPED
        self.error: str | None = None
        self.updated_at = time()
        self._lock = asyncio.Lock()

    def snapshot(self) -> SlotSnapshot:
        return SlotSnapshot(
            name=self.slot.name,
            status=self.status.value,
            user=self.slot.user,
            tmux_session=self.slot.tmux_session,
            process_id=self.endpoint.process_id if self.endpoint else None,
            pipewire_node_id=self.endpoint.pipewire_node_id if self.endpoint else None,
            pipewire_client_id=self.endpoint.pipewire_client_id if self.endpoint else None,
            gamescope_socket=self.endpoint.gamescope_socket if self.endpoint else None,
            eis_socket=self.endpoint.eis_socket if self.endpoint else None,
            error=self.error,
            updated_at=self.updated_at,
        )

    def _apply_snapshot(self, snapshot: dict) -> None:
        self.status = SlotStatus(snapshot["status"])
        self.error = snapshot.get("error")
        if snapshot.get("process_id") is not None and snapshot.get("pipewire_node_id") is not None:
            self.endpoint = DiscoveredEndpoint(
                uid=0,
                pipewire_node_id=snapshot["pipewire_node_id"],
                pipewire_client_id=snapshot["pipewire_client_id"],
                process_id=snapshot["process_id"],
                gamescope_socket=snapshot["gamescope_socket"],
                eis_socket=snapshot["eis_socket"],
            )
        elif snapshot.get("process_id") is not None:
            self.endpoint = DiscoveredEndpoint(
                uid=0,
                pipewire_node_id=snapshot.get("pipewire_node_id") or 0,
                pipewire_client_id=snapshot.get("pipewire_client_id") or 0,
                process_id=snapshot["process_id"],
                gamescope_socket=snapshot.get("gamescope_socket") or "",
                eis_socket=snapshot.get("eis_socket") or "",
            )
        else:
            self.endpoint = None
        self.updated_at = snapshot.get("updated_at", time())

    async def _poll_agent_snapshot(self, timeout_s: float) -> dict:
        deadline = time() + timeout_s
        last_error = None
        while time() < deadline:
            try:
                response = await asyncio.to_thread(request_status, self.slot, 1.0)
                if not response.get("ok"):
                    last_error = response.get("error")
                else:
                    snapshot = response["snapshot"]
                    self._apply_snapshot(snapshot)
                    if snapshot["status"] == SlotStatus.READY.value:
                        return snapshot
                    if snapshot["status"] == SlotStatus.ERROR.value:
                        return snapshot
            except Exception as exc:
                last_error = str(exc)
            await asyncio.sleep(self.harness.runtime.discovery_poll_interval_s)
        raise TimeoutError(last_error or f"slot {self.slot.name} agent did not respond")

    async def start(self) -> SlotSnapshot:
        async with self._lock:
            self.status = SlotStatus.LAUNCHING
            self.error = None
            self.updated_at = time()
            try:
                await asyncio.to_thread(launch_slot_session, self.harness, self.slot)
                snapshot = await self._poll_agent_snapshot(
                    self.slot.startup_grace_s + self.harness.runtime.discovery_timeout_s
                )
                if snapshot["status"] == SlotStatus.ERROR.value:
                    self.status = SlotStatus.ERROR
                    self.error = snapshot.get("error")
            except Exception as exc:
                self.status = SlotStatus.ERROR
                self.error = str(exc)
            self.updated_at = time()
            return self.snapshot()

    async def stop(self) -> SlotSnapshot:
        async with self._lock:
            self.status = SlotStatus.STOPPING
            self.updated_at = time()
            try:
                try:
                    await asyncio.to_thread(request_stop, self.slot, 1.0)
                    await asyncio.sleep(0.5)
                except Exception:
                    pass
                await asyncio.to_thread(tmux_kill_session, self.slot.tmux_session)
                self.endpoint = None
                self.status = SlotStatus.STOPPED
                self.error = None
            except Exception as exc:
                self.status = SlotStatus.ERROR
                self.error = str(exc)
            self.updated_at = time()
            return self.snapshot()

    async def restart(self) -> SlotSnapshot:
        await self.stop()
        await asyncio.sleep(self.harness.runtime.restart_backoff_s)
        return await self.start()

    async def latest_jpeg(self) -> bytes | None:
        try:
            return await asyncio.to_thread(request_frame, self.slot, 1.0)
        except Exception:
            return None

    async def apply_input_event(self, payload: dict) -> None:
        response = await asyncio.to_thread(send_input, self.slot, payload, 1.0)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", f"slot {self.slot.name} input failed"))
