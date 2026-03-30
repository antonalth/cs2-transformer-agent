from __future__ import annotations

from dataclasses import asdict
import asyncio
from time import time

from .compositor import CompositeRenderer
from .config import HarnessConfig
from .models import SlotStatus
from .process_control import cleanup_global_source_locks, cleanup_slot_processes
from .remote_protocol import encode_observation, new_metadata, observation_metadata_item
from .worker import SlotWorker


class HarnessSupervisor:
    def __init__(self, config: HarnessConfig) -> None:
        self.config = config
        self.workers = {slot.name: SlotWorker(config, slot) for slot in config.slots}
        self.compositor = CompositeRenderer()

    async def start(self) -> None:
        if self.config.web.auto_start_slots:
            await asyncio.gather(*(worker.start() for worker in self.workers.values()))

    async def stop(self) -> None:
        await asyncio.gather(*(worker.stop() for worker in self.workers.values()), return_exceptions=True)

    async def cleanup_all(self) -> dict:
        await asyncio.gather(*(worker.stop() for worker in self.workers.values()), return_exceptions=True)
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

    async def send_input(self, slot_name: str, payload: dict) -> None:
        worker = self.get_worker(slot_name)
        await worker.apply_input_event(payload)

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
