from __future__ import annotations

import argparse

from .config import Model3RuntimeConfig
from .runtime import Model3InferenceRuntime
from .webapp import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--harness-url", required=True)
    parser.add_argument("--data-root", default="./dataset0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--transport", choices=("http", "stream"), default="http")
    parser.add_argument("--poll-interval", type=float, default=1.0 / 32.0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--cache-window-frames", type=int, default=None)
    parser.add_argument("--browser-fps", type=float, default=2.0)
    parser.add_argument("--recording-fps", type=float, default=10.0)
    parser.add_argument("--insecure", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "uvicorn is not installed. Install inference requirements first: "
            "pip install -r inference/requirements-inference.txt"
        ) from exc

    cfg = Model3RuntimeConfig(
        checkpoint_path=args.checkpoint,
        harness_url=args.harness_url,
        data_root=args.data_root,
        device=args.device,
        verify_ssl=not args.insecure,
        transport=args.transport,
        poll_interval_s=args.poll_interval,
        cache_window_frames=args.cache_window_frames,
        browser_fps=args.browser_fps,
        recording_fps=args.recording_fps,
    )
    runtime = Model3InferenceRuntime(cfg)
    app = create_app(runtime)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
