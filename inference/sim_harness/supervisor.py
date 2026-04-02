from __future__ import annotations

from dataclasses import asdict
import asyncio
from time import time

from .compositor import CompositeRenderer
from .config import HarnessConfig
from .models import SlotStatus
from .process_control import cleanup_global_source_locks, cleanup_slot_processes
from .server_worker import ServerWorker
from .remote_protocol import encode_observation, new_metadata, observation_metadata_item
from .worker import SlotWorker


TYPEABLE_CHAR_MAP: dict[str, tuple[str, bool]] = {
    "a": ("a", False),
    "b": ("b", False),
    "c": ("c", False),
    "d": ("d", False),
    "e": ("e", False),
    "f": ("f", False),
    "g": ("g", False),
    "h": ("h", False),
    "i": ("i", False),
    "j": ("j", False),
    "k": ("k", False),
    "l": ("l", False),
    "m": ("m", False),
    "n": ("n", False),
    "o": ("o", False),
    "p": ("p", False),
    "q": ("q", False),
    "r": ("r", False),
    "s": ("s", False),
    "t": ("t", False),
    "u": ("u", False),
    "v": ("v", False),
    "w": ("w", False),
    "x": ("x", False),
    "y": ("y", False),
    "z": ("z", False),
    "A": ("a", True),
    "B": ("b", True),
    "C": ("c", True),
    "D": ("d", True),
    "E": ("e", True),
    "F": ("f", True),
    "G": ("g", True),
    "H": ("h", True),
    "I": ("i", True),
    "J": ("j", True),
    "K": ("k", True),
    "L": ("l", True),
    "M": ("m", True),
    "N": ("n", True),
    "O": ("o", True),
    "P": ("p", True),
    "Q": ("q", True),
    "R": ("r", True),
    "S": ("s", True),
    "T": ("t", True),
    "U": ("u", True),
    "V": ("v", True),
    "W": ("w", True),
    "X": ("x", True),
    "Y": ("y", True),
    "Z": ("z", True),
    "0": ("0", False),
    "1": ("1", False),
    "2": ("2", False),
    "3": ("3", False),
    "4": ("4", False),
    "5": ("5", False),
    "6": ("6", False),
    "7": ("7", False),
    "8": ("8", False),
    "9": ("9", False),
    " ": ("space", False),
    ".": ("period", False),
    ":": ("semicolon", True),
    "-": ("minus", False),
    "_": ("minus", True),
    "/": ("slash", False),
}


class HarnessSupervisor:
    def __init__(self, config: HarnessConfig) -> None:
        self.config = config
        self.workers = {slot.name: SlotWorker(config, slot) for slot in config.slots}
        self.server = ServerWorker(config)
        self.compositor = CompositeRenderer()

    async def start(self) -> None:
        if self.config.web.auto_start_slots:
            await asyncio.gather(*(worker.start() for worker in self.workers.values()))

    async def stop(self) -> None:
        await asyncio.gather(*(worker.stop() for worker in self.workers.values()), return_exceptions=True)
        await self.server.stop()

    async def cleanup_all(self) -> dict:
        await asyncio.gather(*(worker.stop() for worker in self.workers.values()), return_exceptions=True)
        await self.server.stop()
        cleanup = await asyncio.gather(
            *(asyncio.to_thread(cleanup_slot_processes, self.config, worker.slot) for worker in self.workers.values())
        )
        removed_global_locks = await asyncio.to_thread(cleanup_global_source_locks, self.config)
        for worker in self.workers.values():
            worker.endpoint = None
            worker.status = SlotStatus.STOPPED
            worker.error = None
            worker.updated_at = time()
        return {
            "ok": True,
            "cleanup": cleanup,
            "removed_global_locks": removed_global_locks,
            "slots": self.list_slots(),
            "server": self.server_snapshot(),
        }

    def list_slots(self) -> list[dict]:
        return [asdict(worker.snapshot()) for worker in self.workers.values()]

    def get_worker(self, slot_name: str) -> SlotWorker:
        try:
            return self.workers[slot_name]
        except KeyError as exc:
            raise KeyError(f"unknown slot: {slot_name}") from exc

    async def start_slot(self, slot_name: str) -> dict:
        worker = self.get_worker(slot_name)
        return asdict(await worker.start())

    async def stop_slot(self, slot_name: str) -> dict:
        worker = self.get_worker(slot_name)
        return asdict(await worker.stop())

    async def restart_slot(self, slot_name: str) -> dict:
        worker = self.get_worker(slot_name)
        return asdict(await worker.restart())

    def server_snapshot(self) -> dict:
        return asdict(self.server.snapshot())

    async def start_server(self) -> dict:
        return asdict(await self.server.start())

    async def stop_server(self) -> dict:
        return asdict(await self.server.stop())

    async def restart_server(self) -> dict:
        return asdict(await self.server.restart())

    async def server_logs(self, lines: int | None = None) -> dict:
        return await self.server.read_logs(lines)

    async def send_server_command(self, command: str) -> dict:
        return await self.server.send_command(command)

    async def pause_server(self) -> dict:
        command = self.config.server.pause_command.strip()
        if not command:
            raise RuntimeError("server pause_command is empty")
        result = await self.server.send_command(command)
        result["operation"] = "pause"
        return result

    async def resume_server(self) -> dict:
        command = self.config.server.resume_command.strip()
        if not command:
            raise RuntimeError("server resume_command is empty")
        result = await self.server.send_command(command)
        result["operation"] = "resume"
        return result

    async def list_server_scenarios(self) -> dict:
        return await self.server.list_scenarios()

    async def server_plugin_state(self, *, refresh: bool = True) -> dict:
        return await self.server.plugin_state(refresh=refresh)

    async def run_server_scenario(self, scenario_name: str, *, op: str) -> dict:
        return await self.server.run_scenario_command(scenario_name, op=op)

    async def send_input(self, slot_name: str, payload: dict) -> None:
        worker = self.get_worker(slot_name)
        await worker.apply_input_event(payload)

    async def connect_slot_to_server(self, slot_name: str, *, address: str | None = None) -> dict:
        worker = self.get_worker(slot_name)
        target = (address or self.config.server.connect_address).strip()
        if not target:
            raise RuntimeError("server connect address is empty")

        toggle_key = self.config.server.console_toggle_key
        toggle_with_shift = self.config.server.console_toggle_with_shift
        submit_key = self.config.server.connect_submit_key
        close_key = self.config.server.console_close_key
        key_delay = max(0.0, self.config.server.connect_keystroke_delay_s)
        if toggle_with_shift:
            await worker.apply_input_event({"t": "key", "key": "lshift", "down": True})
        await worker.apply_input_event({"t": "key", "key": toggle_key, "down": True})
        await worker.apply_input_event({"t": "key", "key": toggle_key, "down": False})
        if toggle_with_shift:
            await worker.apply_input_event({"t": "key", "key": "lshift", "down": False})
        await asyncio.sleep(max(0.0, self.config.server.connect_open_delay_s))

        connect_command = self.config.server.connect_command_template.format(address=target)
        for ch in connect_command:
            try:
                key_name, use_shift = TYPEABLE_CHAR_MAP[ch]
            except KeyError as exc:
                raise RuntimeError(f"unsupported character for connect command: {ch!r}") from exc
            if use_shift:
                await worker.apply_input_event({"t": "key", "key": "lshift", "down": True})
            await worker.apply_input_event({"t": "key", "key": key_name, "down": True})
            await worker.apply_input_event({"t": "key", "key": key_name, "down": False})
            if use_shift:
                await worker.apply_input_event({"t": "key", "key": "lshift", "down": False})
            if key_delay > 0.0:
                await asyncio.sleep(key_delay)

        await worker.apply_input_event({"t": "key", "key": submit_key, "down": True})
        await worker.apply_input_event({"t": "key", "key": submit_key, "down": False})
        await asyncio.sleep(max(0.0, self.config.server.connect_post_enter_delay_s))
        await worker.apply_input_event({"t": "key", "key": close_key, "down": True})
        await worker.apply_input_event({"t": "key", "key": close_key, "down": False})
        return {"ok": True, "slot": slot_name, "address": target}

    async def connect_all_slots_to_server(self, *, address: str | None = None) -> dict:
        results: dict[str, dict] = {}
        for slot_name in self.workers:
            try:
                results[slot_name] = await self.connect_slot_to_server(slot_name, address=address)
            except Exception as exc:
                results[slot_name] = {"ok": False, "error": str(exc), "slot": slot_name}
        return {"ok": all(item.get("ok") for item in results.values()), "results": results}

    async def slot_audio_pcm(self, slot_name: str) -> tuple[int, int, bytes] | None:
        worker = self.get_worker(slot_name)
        return await worker.latest_audio_pcm()

    async def observation_payload(self) -> bytes:
        workers = list(self.workers.values())
        snapshots = [worker.snapshot() for worker in workers]
        frame_packets = await asyncio.gather(*(worker.latest_frame_packet() for worker in workers))
        audio_packets = await asyncio.gather(*(worker.latest_audio_pcm() for worker in workers))
        metadata = new_metadata()
        blobs: list[bytes] = []
        for worker, snapshot, frame_packet, audio_packet in zip(workers, snapshots, frame_packets, audio_packets, strict=True):
            frame_seq = 0
            frame_time_ns = 0
            frame_blob = b""
            if frame_packet is not None:
                frame_seq, frame_time_ns, frame_blob = frame_packet

            audio_seq = 0
            audio_time_ns = 0
            audio_blob = b""
            if audio_packet is not None:
                audio_seq, audio_time_ns, audio_blob = audio_packet

            metadata["slots"].append(
                observation_metadata_item(
                    name=worker.slot.name,
                    status=snapshot.status,
                    error=snapshot.error,
                    frame_seq=frame_seq,
                    frame_time_ns=frame_time_ns,
                    frame_size=len(frame_blob),
                    audio_seq=audio_seq,
                    audio_time_ns=audio_time_ns,
                    audio_size=len(audio_blob),
                    audio_sample_rate=worker.slot.audio_sample_rate,
                    audio_channels=worker.slot.audio_channels,
                )
            )
            blobs.append(frame_blob)
            blobs.append(audio_blob)
        return encode_observation(metadata, blobs)

    async def apply_actions(self, actions: dict[str, list[dict] | dict]) -> dict:
        results: dict[str, dict] = {}
        for slot_name, payload in actions.items():
            worker = self.get_worker(slot_name)
            events = payload if isinstance(payload, list) else [payload]
            try:
                for event in events:
                    await worker.apply_input_event(event)
                results[slot_name] = {"ok": True, "count": len(events)}
            except Exception as exc:
                results[slot_name] = {"ok": False, "error": str(exc)}
        return results

    async def render_composite(self) -> bytes:
        workers = list(self.workers.values())
        snapshots = [worker.snapshot() for worker in workers]
        frames = await asyncio.gather(*(worker.latest_jpeg() for worker in workers))
        return self.compositor.render_jpeg(frames, snapshots)
