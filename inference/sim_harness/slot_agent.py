from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import socket
import struct
import subprocess
from dataclasses import asdict
from time import time

from .capture import SlotCaptureBridge
from .config import load_config
from .discovery import discover_new_endpoint_local, snapshot_endpoint_names_local
from .models import SlotSnapshot, SlotStatus


class SlotAgent:
    def __init__(self, config_path: str, slot_name: str) -> None:
        self.config = load_config(config_path)
        self.slot = self.config.slot_map()[slot_name]
        self.status = SlotStatus.STOPPED
        self.error: str | None = None
        self.endpoint = None
        self.capture: SlotCaptureBridge | None = None
        self.child: subprocess.Popen[str] | None = None
        self.updated_at = time()
        self._stop_event = asyncio.Event()
        self._servers: list[asyncio.base_events.Server] = []

    def snapshot(self) -> dict:
        return asdict(
            SlotSnapshot(
                name=self.slot.name,
                status=self.status.value,
                user=self.slot.user,
                tmux_session=self.slot.tmux_session,
                process_id=self.endpoint.process_id if self.endpoint else (self.child.pid if self.child else None),
                pipewire_node_id=self.endpoint.pipewire_node_id if self.endpoint else None,
                pipewire_client_id=self.endpoint.pipewire_client_id if self.endpoint else None,
                gamescope_socket=self.endpoint.gamescope_socket if self.endpoint else None,
                eis_socket=self.endpoint.eis_socket if self.endpoint else None,
                error=self.error,
                updated_at=self.updated_at,
            )
        )

    def _set_status(self, status: SlotStatus, error: str | None = None) -> None:
        self.status = status
        self.error = error
        self.updated_at = time()

    async def start(self) -> None:
        self._set_status(SlotStatus.LAUNCHING)
        prev_eis, prev_nodes = snapshot_endpoint_names_local(self.slot.user)
        self.child = subprocess.Popen(
            [
                "gamescope",
                *self.slot.effective_gamescope_args(),
                "--",
                *self.slot.launch_command,
            ],
            text=True,
        )
        await asyncio.sleep(self.slot.startup_grace_s)
        self._set_status(SlotStatus.DISCOVERING)
        try:
            self.endpoint = await asyncio.to_thread(
                discover_new_endpoint_local,
                self.slot.user,
                prev_eis,
                prev_nodes,
                self.config.runtime.discovery_timeout_s,
                self.config.runtime.discovery_poll_interval_s,
            )
            self.capture = SlotCaptureBridge(self.slot, self.endpoint)
            await asyncio.to_thread(self.capture.start)
            self._set_status(SlotStatus.READY)
        except Exception as exc:
            self._set_status(SlotStatus.ERROR, str(exc))

    async def shutdown(self) -> None:
        self._set_status(SlotStatus.STOPPING)
        if self.capture is not None:
            await asyncio.to_thread(self.capture.stop)
            self.capture = None
        if self.child is not None and self.child.poll() is None:
            self.child.terminate()
            try:
                await asyncio.wait_for(asyncio.to_thread(self.child.wait), timeout=5.0)
            except asyncio.TimeoutError:
                self.child.kill()
                await asyncio.to_thread(self.child.wait)
        self._set_status(SlotStatus.STOPPED)
        self._stop_event.set()

    def _refresh_child_state(self) -> None:
        if self.child is None:
            return
        code = self.child.poll()
        if code is None:
            return
        if self.status not in {SlotStatus.STOPPED, SlotStatus.STOPPING, SlotStatus.ERROR}:
            self._set_status(SlotStatus.ERROR, f"game process exited with code {code}")

    async def handle_control(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            if not line:
                return
            payload = json.loads(line.decode("utf-8"))
            op = payload.get("op")
            self._refresh_child_state()

            if op == "status":
                response = {"ok": True, "snapshot": self.snapshot()}
            elif op == "input":
                if self.capture is None or self.status != SlotStatus.READY:
                    raise RuntimeError(f"slot {self.slot.name} is not ready")
                await asyncio.to_thread(self.capture.apply_input_event, payload["event"])
                response = {"ok": True}
            elif op == "stop":
                response = {"ok": True}
                asyncio.create_task(self.shutdown())
            else:
                response = {"ok": False, "error": f"unsupported op: {op}"}
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}

        writer.write((json.dumps(response) + "\n").encode("utf-8"))
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def handle_frame(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            self._refresh_child_state()
            item = None
            if self.capture is not None and self.status == SlotStatus.READY:
                item = await asyncio.to_thread(self.capture.latest_frame_packet)
            if item is None:
                writer.write(struct.pack("!QQI", 0, 0, 0))
            else:
                seq, ts_ns, payload = item
                writer.write(struct.pack("!QQI", seq, ts_ns, len(payload)))
                writer.write(payload)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def handle_audio(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            self._refresh_child_state()
            item = None
            if self.capture is not None and self.status == SlotStatus.READY:
                item = await asyncio.to_thread(self.capture.latest_audio_pcm)
            if item is None:
                writer.write(struct.pack("!QQI", 0, 0, 0))
            else:
                seq, ts_ns, payload = item
                writer.write(struct.pack("!QQI", seq, ts_ns, len(payload)))
                writer.write(payload)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def serve(self) -> None:
        control_server = await asyncio.start_server(self.handle_control, "127.0.0.1", self.slot.input_port)
        frame_server = await asyncio.start_server(self.handle_frame, "127.0.0.1", self.slot.video_port)
        audio_server = await asyncio.start_server(self.handle_audio, "127.0.0.1", self.slot.audio_port)
        self._servers = [control_server, frame_server, audio_server]
        await self.start()
        await self._stop_event.wait()
        for server in self._servers:
            server.close()
            await server.wait_closed()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--slot", required=True)
    return parser.parse_args()


async def _main_async() -> None:
    args = parse_args()
    agent = SlotAgent(args.config, args.slot)

    def _handle_signal(signum, frame) -> None:
        asyncio.get_running_loop().create_task(agent.shutdown())

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    await agent.serve()


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
