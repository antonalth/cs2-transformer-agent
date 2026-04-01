from __future__ import annotations

import argparse
import importlib.util

from .webapp import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="sim_harness.toml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "uvicorn is not installed. Install inference requirements first: "
            "pip install -r requirements-inference.txt"
        ) from exc

    app = create_app(args.config)
    web_cfg = app.state.supervisor.config.web
    kwargs = {
        "host": web_cfg.host,
        "port": web_cfg.port,
        "ws": _select_ws_backend(),
        "ws_ping_interval": None,
        "ws_ping_timeout": None,
    }
    certfile = web_cfg.ssl_certfile.strip()
    keyfile = web_cfg.ssl_keyfile.strip()
    if certfile or keyfile:
        if not certfile or not keyfile:
            raise RuntimeError("both web.ssl_certfile and web.ssl_keyfile must be set for HTTPS")
        kwargs["ssl_certfile"] = certfile
        kwargs["ssl_keyfile"] = keyfile
    uvicorn.run(app, **kwargs)


def _select_ws_backend() -> str:
    # Avoid uvicorn's legacy `websockets` backend on Python 3.14. Prefer wsproto
    # when installed, otherwise use the sans-I/O backend that ships with uvicorn.
    if importlib.util.find_spec("wsproto") is not None:
        return "wsproto"
    return "websockets-sansio"


if __name__ == "__main__":
    main()
