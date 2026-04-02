from __future__ import annotations

from dataclasses import asdict
from time import time
import asyncio

from .config import HarnessConfig
from .models import ServerSnapshot, SlotStatus
from .process_control import (
    launch_server_session,
    read_json_via_runas,
    server_session_running,
    stop_server_session,
    tmux_capture_pane,
    tmux_send_keys,
)


class ServerWorker:
    def __init__(self, harness: HarnessConfig) -> None:
        self.harness = harness
        self.status = SlotStatus.STOPPED
        self.error: str | None = None
        self.updated_at = time()
        self._lock = asyncio.Lock()
        self._refresh_status()

    def snapshot(self) -> ServerSnapshot:
        self._refresh_status()
        return ServerSnapshot(
            status=self.status.value,
            user=self.harness.server.user,
            tmux_session=self.harness.server.tmux_session,
            connect_address=self.harness.server.connect_address,
            error=self.error,
            updated_at=self.updated_at,
        )

    def _set_status(self, status: SlotStatus, error: str | None = None) -> None:
        self.status = status
        self.error = error
        self.updated_at = time()

    def _refresh_status(self) -> None:
        if server_session_running(self.harness.server):
            if self.status != SlotStatus.READY or self.error is not None:
                self._set_status(SlotStatus.READY)
            return
        if self.status not in {SlotStatus.ERROR, SlotStatus.LAUNCHING, SlotStatus.STOPPING}:
            self._set_status(SlotStatus.STOPPED)

    async def start(self) -> ServerSnapshot:
        async with self._lock:
            self._set_status(SlotStatus.LAUNCHING)
            try:
                await asyncio.to_thread(launch_server_session, self.harness)
                self._refresh_status()
                if self.status != SlotStatus.READY:
                    self._set_status(SlotStatus.ERROR, "server tmux session did not start")
            except Exception as exc:
                self._set_status(SlotStatus.ERROR, str(exc))
            return self.snapshot()

    async def stop(self) -> ServerSnapshot:
        async with self._lock:
            self._set_status(SlotStatus.STOPPING)
            try:
                await asyncio.to_thread(stop_server_session, self.harness)
                self._set_status(SlotStatus.STOPPED)
            except Exception as exc:
                self._set_status(SlotStatus.ERROR, str(exc))
            return self.snapshot()

    async def restart(self) -> ServerSnapshot:
        await self.stop()
        await asyncio.sleep(self.harness.runtime.restart_backoff_s)
        return await self.start()

    async def read_logs(self, lines: int | None = None) -> dict:
        requested_lines = lines or self.harness.server.log_lines
        logs = await asyncio.to_thread(tmux_capture_pane, self.harness.server.tmux_session, requested_lines)
        snapshot = self.snapshot()
        return {
            "ok": True,
            "snapshot": asdict(snapshot),
            "lines": requested_lines,
            "logs": logs,
        }

    async def send_command(self, command: str) -> dict:
        if not command.strip():
            raise RuntimeError("command must not be empty")
        if not server_session_running(self.harness.server):
            raise RuntimeError("server tmux session is not running")
        await asyncio.to_thread(tmux_send_keys, self.harness.server.tmux_session, command, enter=True)
        self._refresh_status()
        return {
            "ok": True,
            "snapshot": asdict(self.snapshot()),
            "command": command,
        }

    async def list_scenarios(self) -> dict:
        path = self.harness.server.scenario_config_path.strip()
        if not path:
            raise RuntimeError("server scenario_config_path is empty")
        data = await asyncio.to_thread(read_json_via_runas, self.harness, "root", path)
        scenarios_raw = data.get("scenarios", {})
        if not isinstance(scenarios_raw, dict):
            raise RuntimeError("scenario config has invalid scenarios object")
        scenarios = []
        for name, scenario in sorted(scenarios_raw.items(), key=lambda item: item[0].lower()):
            if not isinstance(scenario, dict):
                continue
            scenarios.append(
                {
                    "name": name,
                    "map": scenario.get("map", ""),
                    "controlledTeam": scenario.get("controlledTeam", ""),
                    "botTeam": scenario.get("botTeam", ""),
                    "botCount": scenario.get("botCount", 0),
                    "assignments": len(scenario.get("assignments", []) or []),
                }
            )
        return {
            "ok": True,
            "path": path,
            "defaultScenario": data.get("defaultScenario"),
            "scenarios": scenarios,
        }

    async def plugin_state(self, *, refresh: bool = True) -> dict:
        path = self.harness.server.plugin_state_path.strip()
        if not path:
            raise RuntimeError("server plugin_state_path is empty")
        if refresh:
            await self.send_command("css_sim_write_state")
            await asyncio.sleep(max(0.0, float(self.harness.server.plugin_state_refresh_delay_s)))
        data = await asyncio.to_thread(read_json_via_runas, self.harness, self.harness.server.user, path)
        return {
            "ok": True,
            "path": path,
            "state": data,
        }

    async def run_scenario_command(self, scenario_name: str, *, op: str) -> dict:
        if op not in {"reset", "apply"}:
            raise RuntimeError(f"unsupported scenario op: {op}")
        scenario = scenario_name.strip()
        if not scenario:
            raise RuntimeError("scenario name must not be empty")
        command = f"css_sim_{op} {scenario}"
        result = await self.send_command(command)
        result["scenario"] = scenario
        result["operation"] = op
        return result
