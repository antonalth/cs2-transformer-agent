from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pwd
import re
import subprocess
import time

from .models import DiscoveredEndpoint


_ID_RE = re.compile(r"^\s*id\s+(\d+),")
_KV_RE = re.compile(r"^\s*(?:\*\s+)?([A-Za-z0-9._-]+)\s*=\s*\"?(.*?)\"?\s*$")


@dataclass(slots=True)
class PipeWireNode:
    node_id: int
    client_id: int
    media_class: str
    node_name: str


@dataclass(slots=True)
class PipeWireClient:
    client_id: int
    process_id: int
    uid: int
    user: str | None


def _run_text(args: list[str]) -> str:
    try:
        proc = subprocess.run(args, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr or stdout or str(exc)
        raise RuntimeError(detail) from exc
    return proc.stdout


def _run_as_user_text(runas_path: str, user: str, args: list[str]) -> str:
    return _run_text(["sudo", "-n", runas_path, user, *args])


def uid_for_user(user: str) -> int:
    return pwd.getpwnam(user).pw_uid


def runtime_dir_for_uid(uid: int) -> Path:
    return Path("/run/user") / str(uid)


def _list_gamescope_socket_names_local(uid: int) -> list[str]:
    output = _run_text(
        [
            "find",
            str(runtime_dir_for_uid(uid)),
            "-maxdepth",
            "1",
            "-type",
            "s",
            "-name",
            "gamescope*",
            "-printf",
            "%f\n",
        ]
    )
    return sorted(line.strip() for line in output.splitlines() if line.strip())


def _list_gamescope_socket_names(runas_path: str, user: str, uid: int) -> list[str]:
    output = _run_as_user_text(
        runas_path,
        user,
        [
            "find",
            str(runtime_dir_for_uid(uid)),
            "-maxdepth",
            "1",
            "-type",
            "s",
            "-name",
            "gamescope*",
            "-printf",
            "%f\n",
        ],
    )
    return sorted(line.strip() for line in output.splitlines() if line.strip())


def find_gamescope_eis_paths(runas_path: str, user: str, uid: int) -> list[Path]:
    return [
        runtime_dir_for_uid(uid) / name
        for name in _list_gamescope_socket_names(runas_path, user, uid)
        if name.endswith("-ei")
    ]


def find_gamescope_socket_paths(runas_path: str, user: str, uid: int) -> list[Path]:
    sockets: list[Path] = []
    for name in _list_gamescope_socket_names(runas_path, user, uid):
        if name.endswith("-ei"):
            continue
        path = runtime_dir_for_uid(uid) / name
        sockets.append(path)
    return sockets


def parse_wpctl_inspect(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        match = _KV_RE.match(line)
        if match:
            out[match.group(1)] = match.group(2)
    return out


def list_gamescope_pipewire_nodes_local() -> list[PipeWireNode]:
    output = _run_text(["pw-cli", "ls", "Node"])
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in output.splitlines():
        if line.startswith("\tid ") or line.startswith("id "):
            if current:
                blocks.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append(current)

    nodes: list[PipeWireNode] = []
    for block in blocks:
        id_match = _ID_RE.search(block[0])
        if not id_match:
            continue
        node_id = int(id_match.group(1))
        meta = parse_wpctl_inspect(_run_text(["wpctl", "inspect", str(node_id)]))
        if meta.get("node.name") != "gamescope":
            continue
        if meta.get("media.class") != "Video/Source":
            continue
        nodes.append(
            PipeWireNode(
                node_id=node_id,
                client_id=int(meta["client.id"]),
                media_class=meta["media.class"],
                node_name=meta["node.name"],
            )
        )
    return sorted(nodes, key=lambda item: item.node_id)


def list_gamescope_pipewire_nodes(runas_path: str, user: str) -> list[PipeWireNode]:
    output = _run_as_user_text(runas_path, user, ["pw-cli", "ls", "Node"])
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in output.splitlines():
        if line.startswith("\tid ") or line.startswith("id "):
            if current:
                blocks.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append(current)

    nodes: list[PipeWireNode] = []
    for block in blocks:
        id_match = _ID_RE.search(block[0])
        if not id_match:
            continue
        node_id = int(id_match.group(1))
        meta = parse_wpctl_inspect(_run_as_user_text(runas_path, user, ["wpctl", "inspect", str(node_id)]))
        if meta.get("node.name") != "gamescope":
            continue
        if meta.get("media.class") != "Video/Source":
            continue
        client_id = int(meta["client.id"])
        nodes.append(
            PipeWireNode(
                node_id=node_id,
                client_id=client_id,
                media_class=meta["media.class"],
                node_name=meta["node.name"],
            )
        )
    return sorted(nodes, key=lambda item: item.node_id)


def inspect_pipewire_client_local(client_id: int) -> PipeWireClient:
    meta = parse_wpctl_inspect(_run_text(["wpctl", "inspect", str(client_id)]))
    return PipeWireClient(
        client_id=client_id,
        process_id=int(meta["application.process.id"]),
        uid=int(meta["pipewire.sec.uid"]),
        user=meta.get("application.process.user"),
    )


def inspect_pipewire_client(runas_path: str, user: str, client_id: int) -> PipeWireClient:
    meta = parse_wpctl_inspect(_run_as_user_text(runas_path, user, ["wpctl", "inspect", str(client_id)]))
    return PipeWireClient(
        client_id=client_id,
        process_id=int(meta["application.process.id"]),
        uid=int(meta["pipewire.sec.uid"]),
        user=meta.get("application.process.user"),
    )


def snapshot_endpoint_names(runas_path: str, user: str) -> tuple[set[str], set[int]]:
    uid = uid_for_user(user)
    eis_names = {path.name for path in find_gamescope_eis_paths(runas_path, user, uid)}
    node_ids = {
        node.node_id
        for node in list_gamescope_pipewire_nodes(runas_path, user)
        if inspect_pipewire_client(runas_path, user, node.client_id).uid == uid
    }
    return eis_names, node_ids


def snapshot_endpoint_names_local(user: str) -> tuple[set[str], set[int]]:
    uid = uid_for_user(user)
    eis_names = {
        (runtime_dir_for_uid(uid) / name).name
        for name in _list_gamescope_socket_names_local(uid)
        if name.endswith("-ei")
    }
    node_ids = {
        node.node_id
        for node in list_gamescope_pipewire_nodes_local()
        if inspect_pipewire_client_local(node.client_id).uid == uid
    }
    return eis_names, node_ids


def discover_new_endpoint(
    runas_path: str,
    user: str,
    previous_eis_names: set[str],
    previous_node_ids: set[int],
    timeout_s: float,
    poll_interval_s: float,
) -> DiscoveredEndpoint:
    uid = uid_for_user(user)
    deadline = time.monotonic() + timeout_s
    last_error = "discovery timed out"
    while time.monotonic() < deadline:
        eis_paths = find_gamescope_eis_paths(runas_path, user, uid)
        new_eis = [path for path in eis_paths if path.name not in previous_eis_names]
        nodes = list_gamescope_pipewire_nodes(runas_path, user)
        new_nodes = []
        for node in nodes:
            if node.node_id in previous_node_ids:
                continue
            client = inspect_pipewire_client(runas_path, user, node.client_id)
            if client.uid == uid:
                new_nodes.append((node, client))

        if new_eis and new_nodes:
            eis_path = new_eis[0]
            socket_name = eis_path.name.removesuffix("-ei")
            gamescope_socket = runtime_dir_for_uid(uid) / socket_name
            node, client = sorted(new_nodes, key=lambda item: item[0].node_id)[0]
            return DiscoveredEndpoint(
                uid=uid,
                pipewire_node_id=node.node_id,
                pipewire_client_id=node.client_id,
                process_id=client.process_id,
                gamescope_socket=str(gamescope_socket),
                eis_socket=str(eis_path),
            )

        current_nodes = []
        for node in nodes:
            client = inspect_pipewire_client(runas_path, user, node.client_id)
            if client.uid == uid:
                current_nodes.append((node, client))

        if len(eis_paths) == 1 and len(current_nodes) == 1:
            eis_path = eis_paths[0]
            socket_name = eis_path.name.removesuffix("-ei")
            gamescope_socket = runtime_dir_for_uid(uid) / socket_name
            node, client = current_nodes[0]
            return DiscoveredEndpoint(
                uid=uid,
                pipewire_node_id=node.node_id,
                pipewire_client_id=node.client_id,
                process_id=client.process_id,
                gamescope_socket=str(gamescope_socket),
                eis_socket=str(eis_path),
            )

        if not eis_paths:
            last_error = f"no EIS socket in {runtime_dir_for_uid(uid)}"
        elif not current_nodes:
            last_error = f"no PipeWire gamescope node visible for user {user}"
        else:
            last_error = (
                f"ambiguous discovery for user {user}: "
                f"{len(eis_paths)} EIS sockets, {len(current_nodes)} PipeWire nodes"
            )
        time.sleep(poll_interval_s)

    raise TimeoutError(last_error)


def discover_new_endpoint_local(
    user: str,
    previous_eis_names: set[str],
    previous_node_ids: set[int],
    timeout_s: float,
    poll_interval_s: float,
) -> DiscoveredEndpoint:
    uid = uid_for_user(user)
    deadline = time.monotonic() + timeout_s
    last_error = "discovery timed out"
    while time.monotonic() < deadline:
        eis_paths = [
            runtime_dir_for_uid(uid) / name
            for name in _list_gamescope_socket_names_local(uid)
            if name.endswith("-ei")
        ]
        new_eis = [path for path in eis_paths if path.name not in previous_eis_names]
        nodes = list_gamescope_pipewire_nodes_local()
        new_nodes = []
        for node in nodes:
            if node.node_id in previous_node_ids:
                continue
            client = inspect_pipewire_client_local(node.client_id)
            if client.uid == uid:
                new_nodes.append((node, client))

        if new_eis and new_nodes:
            eis_path = new_eis[0]
            socket_name = eis_path.name.removesuffix("-ei")
            gamescope_socket = runtime_dir_for_uid(uid) / socket_name
            node, client = sorted(new_nodes, key=lambda item: item[0].node_id)[0]
            return DiscoveredEndpoint(
                uid=uid,
                pipewire_node_id=node.node_id,
                pipewire_client_id=node.client_id,
                process_id=client.process_id,
                gamescope_socket=str(gamescope_socket),
                eis_socket=str(eis_path),
            )

        current_nodes = []
        for node in nodes:
            client = inspect_pipewire_client_local(node.client_id)
            if client.uid == uid:
                current_nodes.append((node, client))

        if len(eis_paths) == 1 and len(current_nodes) == 1:
            eis_path = eis_paths[0]
            socket_name = eis_path.name.removesuffix("-ei")
            gamescope_socket = runtime_dir_for_uid(uid) / socket_name
            node, client = current_nodes[0]
            return DiscoveredEndpoint(
                uid=uid,
                pipewire_node_id=node.node_id,
                pipewire_client_id=node.client_id,
                process_id=client.process_id,
                gamescope_socket=str(gamescope_socket),
                eis_socket=str(eis_path),
            )

        if not eis_paths:
            last_error = f"no EIS socket in {runtime_dir_for_uid(uid)}"
        elif not current_nodes:
            last_error = f"no PipeWire gamescope node visible for user {user}"
        else:
            last_error = (
                f"ambiguous discovery for user {user}: "
                f"{len(eis_paths)} EIS sockets, {len(current_nodes)} PipeWire nodes"
            )
        time.sleep(poll_interval_s)

    raise TimeoutError(last_error)
