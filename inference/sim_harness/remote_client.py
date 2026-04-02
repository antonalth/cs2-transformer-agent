from __future__ import annotations

from dataclasses import dataclass, field
import ssl
import time
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx

from .remote_protocol import (
    HarnessObservation,
    StreamMessage,
    decode_observation,
    decode_stream_text,
    encode_stream_actions,
)


@dataclass(slots=True)
class RemoteHarnessClient:
    base_url: str
    verify_ssl: bool = True
    timeout_s: float = 5.0
    _client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            verify=self.verify_ssl,
            timeout=self.timeout_s,
            headers={"connection": "keep-alive"},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> RemoteHarnessClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def get_slots(self) -> list[dict[str, Any]]:
        return self.get_json("/api/slots")

    def get_observation(self) -> HarnessObservation:
        response = self._client.get("/api/model/observation.bin")
        response.raise_for_status()
        return decode_observation(response.content)

    def send_actions(self, actions: dict[str, list[dict[str, Any]] | dict[str, Any]]) -> dict[str, Any]:
        return self.post_json("/api/model/actions", {"actions": actions})["results"]

    def get_json(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        response = self._client.get(path, params=params)
        response.raise_for_status()
        return response.json()

    def post_json(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        response = self._client.post(path, json=payload or {})
        response.raise_for_status()
        return response.json()


@dataclass(slots=True)
class RemoteHarnessStreamClient:
    base_url: str
    verify_ssl: bool = True
    open_timeout_s: float = 10.0
    ping_interval_s: float = 20.0
    ws_url: str = field(init=False)
    candidate_urls: list[str] = field(init=False, repr=False)
    _ws: Any = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self.candidate_urls = self._stream_urls()
        self.ws_url = self.candidate_urls[0]

    def _stream_urls(self) -> list[str]:
        if "://" in self.base_url:
            parsed = urlparse(self.base_url)
            schemes = [
                {
                    "http": "ws",
                    "https": "wss",
                    "ws": "ws",
                    "wss": "wss",
                }.get(parsed.scheme)
            ]
        else:
            parsed = urlparse(f"ws://{self.base_url}")
            schemes = ["wss", "ws"]

        if not schemes or schemes[0] is None:
            raise ValueError(f"unsupported base_url scheme: {parsed.scheme or '<none>'}")
        path = parsed.path.rstrip("/")
        if path.endswith("/api/model/ws"):
            stream_path = path
        else:
            stream_path = f"{path}/api/model/ws" if path else "/api/model/ws"
        return [urlunparse((scheme, parsed.netloc, stream_path, "", "", "")) for scheme in schemes if scheme]

    def _ssl_context(self, ws_url: str):
        if not ws_url.startswith("wss://"):
            return None
        if self.verify_ssl:
            return ssl.create_default_context()
        return ssl._create_unverified_context()

    def _format_connect_error(self, attempts: list[tuple[str, Exception]]) -> RuntimeError:
        if not attempts:
            return RuntimeError("stream connection failed with no attempts")
        parts = [f"{url} -> {type(exc).__name__}: {exc}" for url, exc in attempts]
        hint = ""
        last_exc = attempts[-1][1]
        if isinstance(last_exc, ssl.SSLCertVerificationError):
            hint = " The server is using TLS with an untrusted certificate. Retry with --insecure or verify_ssl=False."
        elif any(isinstance(exc, Exception) and "valid HTTP response" in str(exc) for _, exc in attempts):
            hint = " The server may be speaking HTTPS/WSS while the client is using plain WS/HTTP."
        return RuntimeError("unable to connect to model stream: " + " | ".join(parts) + hint)

    async def connect(self) -> RemoteHarnessStreamClient:
        import websockets
        from websockets import exceptions as ws_exceptions

        attempts: list[tuple[str, Exception]] = []
        for ws_url in self.candidate_urls:
            try:
                self._ws = await websockets.connect(
                    ws_url,
                    open_timeout=self.open_timeout_s,
                    ping_interval=self.ping_interval_s,
                    max_size=None,
                    ssl=self._ssl_context(ws_url),
                )
                self.ws_url = ws_url
                return self
            except ssl.SSLCertVerificationError as exc:
                attempts.append((ws_url, exc))
                break
            except (ws_exceptions.InvalidMessage, ws_exceptions.InvalidStatus, ssl.SSLError, OSError) as exc:
                attempts.append((ws_url, exc))
                continue
        raise self._format_connect_error(attempts)

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def __aenter__(self) -> RemoteHarnessStreamClient:
        return await self.connect()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def recv_message(self) -> StreamMessage:
        if self._ws is None:
            raise RuntimeError("stream client not connected")
        raw = await self._ws.recv()
        if isinstance(raw, str):
            payload = decode_stream_text(raw)
            return StreamMessage(kind=str(payload.get("op", "message")), raw_size=len(raw.encode("utf-8")), payload=payload)
        if isinstance(raw, bytes):
            return StreamMessage(kind="observation", raw_size=len(raw), observation=decode_observation(raw))
        raise TypeError(f"unsupported websocket payload type: {type(raw)!r}")

    async def recv_observation(self) -> HarnessObservation:
        while True:
            message = await self.recv_message()
            if message.observation is not None:
                return message.observation

    async def recv_model3_batch(self, window_builder) -> Any:
        observation = await self.recv_observation()
        if window_builder is None:
            from .model3_helpers import observation_to_model3_batch

            return observation_to_model3_batch(observation)
        while True:
            window_builder.push(observation)
            if getattr(window_builder, "ready", False):
                return window_builder.build_batch()
            observation = await self.recv_observation()

    async def send_actions(
        self,
        actions: dict[str, list[dict[str, Any]] | dict[str, Any]],
        *,
        client_send_ns: int | None = None,
    ) -> int:
        if self._ws is None:
            raise RuntimeError("stream client not connected")
        actual_send_ns = client_send_ns if client_send_ns is not None else time.time_ns()
        payload = encode_stream_actions(actions, client_send_ns=actual_send_ns)
        await self._ws.send(payload)
        return actual_send_ns
