from __future__ import annotations

import argparse
import asyncio
from collections import deque
from dataclasses import dataclass, field
import math
import time

from .remote_client import RemoteHarnessStreamClient
from .remote_protocol import HarnessObservation, StreamMessage


def _mean_ms(values: deque[int]) -> float:
    if not values:
        return float("nan")
    return (sum(values) / len(values)) / 1_000_000.0


def _p95_ms(values: deque[int]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.95) - 1))
    return ordered[idx] / 1_000_000.0


def _fmt_ms(value: float) -> str:
    if not math.isfinite(value):
        return "-"
    return f"{value:.1f}ms"


@dataclass(slots=True)
class StreamStats:
    started_ns: int = field(default_factory=time.time_ns)
    last_report_ns: int = field(default_factory=time.time_ns)
    total_messages: int = 0
    total_bytes: int = 0
    interval_messages: int = 0
    interval_bytes: int = 0
    observation_latency_ns: deque[int] = field(default_factory=lambda: deque(maxlen=512))
    frame_age_ns: deque[int] = field(default_factory=lambda: deque(maxlen=512))
    audio_age_ns: deque[int] = field(default_factory=lambda: deque(maxlen=512))
    action_rtt_ns: deque[int] = field(default_factory=lambda: deque(maxlen=256))
    slot_statuses: dict[str, str] = field(default_factory=dict)
    last_error: str = ""

    def record_observation(self, message: StreamMessage) -> None:
        if message.observation is None:
            return
        now_ns = time.time_ns()
        observation = message.observation
        self.total_messages += 1
        self.total_bytes += message.raw_size
        self.interval_messages += 1
        self.interval_bytes += message.raw_size
        self.observation_latency_ns.append(max(0, now_ns - observation.server_time_ns))
        for slot in observation.slots:
            self.slot_statuses[slot.name] = slot.status
            if slot.frame_time_ns > 0:
                self.frame_age_ns.append(max(0, now_ns - slot.frame_time_ns))
            if slot.audio_time_ns > 0:
                self.audio_age_ns.append(max(0, now_ns - slot.audio_time_ns))

    def record_ack(self, payload: dict) -> None:
        client_send_ns = int(payload.get("client_send_ns") or 0)
        if client_send_ns > 0:
            self.action_rtt_ns.append(max(0, time.time_ns() - client_send_ns))

    def record_error(self, detail: str) -> None:
        self.last_error = detail

    def report(self, final: bool = False) -> str:
        now_ns = time.time_ns()
        interval_s = max(1e-6, (now_ns - self.last_report_ns) / 1_000_000_000.0)
        total_s = max(1e-6, (now_ns - self.started_ns) / 1_000_000_000.0)
        if final:
            obs_hz = self.total_messages / total_s
            mib_s = (self.total_bytes / total_s) / (1024 * 1024)
        else:
            obs_hz = self.interval_messages / interval_s
            mib_s = (self.interval_bytes / interval_s) / (1024 * 1024)
        total_mib = self.total_bytes / (1024 * 1024)
        statuses = " ".join(f"{name}:{status}" for name, status in sorted(self.slot_statuses.items()))
        prefix = "final" if final else "stats"
        line = (
            f"{prefix} t={total_s:.1f}s obs={self.total_messages} hz={obs_hz:.1f} "
            f"throughput={mib_s:.2f}MiB/s total={total_mib:.2f}MiB "
            f"obs_age(avg/p95)={_fmt_ms(_mean_ms(self.observation_latency_ns))}/{_fmt_ms(_p95_ms(self.observation_latency_ns))} "
            f"frame_age(avg/p95)={_fmt_ms(_mean_ms(self.frame_age_ns))}/{_fmt_ms(_p95_ms(self.frame_age_ns))} "
            f"audio_age(avg/p95)={_fmt_ms(_mean_ms(self.audio_age_ns))}/{_fmt_ms(_p95_ms(self.audio_age_ns))} "
            f"action_rtt(avg/p95)={_fmt_ms(_mean_ms(self.action_rtt_ns))}/{_fmt_ms(_p95_ms(self.action_rtt_ns))}"
        )
        if statuses:
            line += f" statuses=[{statuses}]"
        if self.last_error:
            line += f" last_error={self.last_error}"
        self.last_report_ns = now_ns
        self.interval_messages = 0
        self.interval_bytes = 0
        return line


def _action_payload(slot_name: str) -> dict[str, list[dict]]:
    return {
        slot_name: [
            {
                "t": "cursor",
                "x": 0.5,
                "y": 0.5,
                "visible": False,
            }
        ]
    }


async def _recv_loop(client: RemoteHarnessStreamClient, stats: StreamStats, stop: asyncio.Event) -> None:
    while not stop.is_set():
        message = await client.recv_message()
        if message.observation is not None:
            stats.record_observation(message)
            continue
        payload = message.payload or {}
        if message.kind == "actions_ack":
            stats.record_ack(payload)
            continue
        if message.kind == "error":
            stats.record_error(str(payload.get("detail", payload)))


async def _send_loop(
    client: RemoteHarnessStreamClient,
    stop: asyncio.Event,
    *,
    action_slot: str,
    action_interval_s: float,
) -> None:
    while not stop.is_set():
        await client.send_actions(_action_payload(action_slot))
        try:
            await asyncio.wait_for(stop.wait(), timeout=action_interval_s)
        except asyncio.TimeoutError:
            continue


async def _report_loop(stats: StreamStats, stop: asyncio.Event, interval_s: float) -> None:
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            print(stats.report(), flush=True)
            continue


async def _run(args: argparse.Namespace) -> None:
    stats = StreamStats()
    stop = asyncio.Event()
    async with RemoteHarnessStreamClient(args.url, verify_ssl=not args.insecure) as client:
        recv_task = asyncio.create_task(_recv_loop(client, stats, stop))
        report_task = asyncio.create_task(_report_loop(stats, stop, args.report_interval))
        tasks = [recv_task, report_task]
        if args.action_slot:
            tasks.append(
                asyncio.create_task(
                    _send_loop(
                        client,
                        stop,
                        action_slot=args.action_slot,
                        action_interval_s=args.action_interval,
                    )
                )
            )

        try:
            await asyncio.sleep(args.duration)
        finally:
            stop.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            print(stats.report(final=True), flush=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test the sim harness websocket model stream")
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="Harness base URL or ws/wss URL")
    parser.add_argument("--duration", type=float, default=10.0, help="How long to run the stream test")
    parser.add_argument("--report-interval", type=float, default=1.0, help="Stats print interval in seconds")
    parser.add_argument(
        "--action-slot",
        default="",
        help="Optional slot name to exercise the return channel with a harmless cursor event",
    )
    parser.add_argument(
        "--action-interval",
        type=float,
        default=1.0,
        help="How often to send the test action when --action-slot is set",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification for wss/https endpoints",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
