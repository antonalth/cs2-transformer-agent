from __future__ import annotations

import argparse

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
    uvicorn.run(app, host="0.0.0.0", port=app.state.supervisor.config.web.port)


if __name__ == "__main__":
    main()
