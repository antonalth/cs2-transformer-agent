from __future__ import annotations

from dataclasses import asdict
import asyncio

from .compositor import CompositeRenderer
from .config import HarnessConfig
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

    async def render_composite(self) -> bytes:
        workers = list(self.workers.values())
        snapshots = [worker.snapshot() for worker in workers]
        frames = await asyncio.gather(*(worker.latest_jpeg() for worker in workers))
        return self.compositor.render_jpeg(frames, snapshots)
