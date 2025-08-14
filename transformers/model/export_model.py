
#!/usr/bin/env python3
"""
export_model.py
---------------
Utilities to:
  1) instantiate your model (via a user-provided factory),
  2) export to ONNX,
  3) build a TensorRT engine,
  4) benchmark PyTorch vs. ONNX Runtime vs. TensorRT.

Assumptions for inputs/outputs mirror the earlier plan:
- Inputs: images [B,T,P,3,H,W], mel_spectrogram [B,T,P,1,128,M], alive_mask [B,T,P]
- Outputs: (players_flat [B,P,Dp], strategy_flat [B,Ds]) as Tensors

You provide a --factory dotted path that returns a torch.nn.Module compatible with the shim below.
"""
import argparse
import importlib
import os
import sys
import time
from dataclasses import dataclass
from typing import Tuple, Optional

import torch

# Optional deps (handled gracefully)
try:
    import onnx
    import onnxruntime as ort
except Exception:  # pragma: no cover
    onnx = None
    ort = None

try:
    import tensorrt as trt
except Exception:  # pragma: no cover
    trt = None


# ------------------------------
# Shim that flattens outputs
# ------------------------------
class CS2TRTShim(torch.nn.Module):
    """
    Wraps your core model to make ONNX/TensorRT export friendly.
    We expect core(batch_dict) -> nested dict with keys like out["player"], out["game_strategy"].
    Adjust mapping to your exact model outputs if they differ.
    """
    def __init__(self, core: torch.nn.Module):
        super().__init__()
        self.core = core

    def forward(self, images, mel_spectrogram, alive_mask):
        batch = {
            "images": images,                     # [B, T, P, 3, H, W]
            "mel_spectrogram": mel_spectrogram,   # [B, T, P, 1, 128, M]
            "alive_mask": alive_mask,             # [B, T, P]
        }
        out = self.core(batch)  # <- adapt if your forward signature is different

        # ---- Example flattening; change to match your heads exactly ----
        # Expect out["player"] to be an iterable of per-player dicts
        # and out["game_strategy"] to be a dict of strategy tensors.
        player_tensors = []
        for p in out["player"]:
            # You MUST adapt these keys to your real model.
            pieces = []
            for k in ("stats", "mouse", "keyboard", "eco", "inventory", "active_weapon"):
                if k in p:
                    x = p[k]
                    if x.dim() == 1:
                        x = x.unsqueeze(0)
                    pieces.append(x)
            if "pos_heatmap" in p:
                pieces.append(p["pos_heatmap"].flatten(2))
            player_tensors.append(torch.cat(pieces, dim=-1))
        players_flat = torch.stack(player_tensors, dim=1)  # [B, P, Dp]

        gs = out["game_strategy"]
        gs_pieces = []
        for k in ("round_state_logits", "round_number", "enemy_pos_vol"):
            if k in gs:
                x = gs[k]
                if k == "round_number":
                    x = x.unsqueeze(-1)
                if x.dim() > 2:
                    x = x.flatten(1)
                gs_pieces.append(x)
        strategy_flat = torch.cat(gs_pieces, dim=-1)  # [B, Ds]

        return players_flat, strategy_flat


# ------------------------------
# Utilities
# ------------------------------
def load_factory(factory_path: str):
    """
    Import a user-provided factory function that builds the model.
    The factory must return a torch.nn.Module on CUDA, in eval() mode.
    Example: --factory "model:build_model"
    """
    if ":" not in factory_path:
        raise ValueError("Factory must be in the form 'module_path:callable_name'")
    mod_name, fn_name = factory_path.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    return fn


@dataclass
class Shapes:
    B: int = 1
    T: int = 1
    P: int = 5
    H: int = 480
    W: int = 640
    M: int = 6


def make_inputs(shp: Shapes, dtype: torch.dtype, device: str = "cuda"):
    B, T, P, H, W, M = shp.B, shp.T, shp.P, shp.H, shp.W, shp.M
    C = 3
    images = torch.randn(B, T, P, C, H, W, device=device, dtype=dtype)
    mel = torch.randn(B, T, P, 1, 128, M, device=device, dtype=dtype)
    alive = torch.ones(B, T, P, device=device, dtype=torch.bool)
    return images, mel, alive


def export_to_onnx(core_model: torch.nn.Module, onnx_path: str, shp: Shapes, dtype: torch.dtype, opset: int = 18):
    core_model.eval().to("cuda")
    shim = CS2TRTShim(core_model).to("cuda")
    images, mel, alive = make_inputs(shp, dtype)

    dynamic_axes = {
        "images": {0: "B", 1: "T", 2: "P", 4: "H", 5: "W"},
        "mel_spectrogram": {0: "B", 1: "T", 2: "P", 5: "M"},
        "alive_mask": {0: "B", 1: "T", 2: "P"},
        "players_flat": {0: "B", 1: "P"},
        "strategy_flat": {0: "B"},
    }

    torch.onnx.export(
        shim, (images, mel, alive), onnx_path,
        input_names=["images", "mel_spectrogram", "alive_mask"],
        output_names=["players_flat", "strategy_flat"],
        do_constant_folding=True,
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        export_params=True
    )

    if onnx is not None:
        mdl = onnx.load(onnx_path)
        onnx.checker.check_model(mdl)


def build_trt_engine(onnx_path: str, engine_path: str, precision: str = "fp16", workspace_gb: int = 8):
    if trt is None:
        raise RuntimeError("tensorrt is not available. Please install TensorRT.")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("ONNX parse error:", parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))

    p = precision.lower()
    if p in ("fp16", "bf16"):
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
    elif p == "int8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
        else:
            print("Warning: INT8 not supported; falling back to FP16/FP32")

    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("Engine build failed")
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
    return engine_path


class TRTRunner:
    def __init__(self, engine_path: str):
        if trt is None:
            raise RuntimeError("tensorrt is not available.")
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.bind_idx = {self.engine.get_binding_name(i): i for i in range(self.engine.num_bindings)}

    def _infer_dyn(self, shape: Tuple[int, ...], ref: Tuple[int, ...]) -> Tuple[int, ...]:
        if len(shape) != len(ref):
            return ref
        return tuple(r if s == -1 else s for s, r in zip(shape, ref))

    def __call__(self, images: torch.Tensor, mel: torch.Tensor, alive: torch.Tensor):
        assert images.is_cuda and mel.is_cuda and alive.is_cuda

        # Set dynamic shapes
        self.context.set_input_shape("images", tuple(images.shape))
        self.context.set_input_shape("mel_spectrogram", tuple(mel.shape))
        self.context.set_input_shape("alive_mask", tuple(alive.shape))

        # Output shapes (may contain -1 before enqueue)
        pb = self.bind_idx["players_flat"]
        sb = self.bind_idx["strategy_flat"]
        ps = self.engine.get_binding_shape(pb)
        ss = self.engine.get_binding_shape(sb)

        # Fallback guess using batch/player dims from inputs.
        ps = self._infer_dyn(ps, (images.shape[0], images.shape[2], 2048))  # EDIT: set your final Dp
        ss = self._infer_dyn(ss, (images.shape[0], 1024))                  # EDIT: set your final Ds

        players_out = torch.empty(ps, device=images.device, dtype=images.dtype)
        strategy_out = torch.empty(ss, device=images.device, dtype=images.dtype)

        bindings = [None] * self.engine.num_bindings
        def bind(name, tensor):
            bindings[self.bind_idx[name]] = tensor.data_ptr()

        bind("images", images)
        bind("mel_spectrogram", mel)
        bind("alive_mask", alive)
        bind("players_flat", players_out)
        bind("strategy_flat", strategy_out)

        self.context.execute_v2(bindings)
        return players_out, strategy_out


# ------------------------------
# Benchmarks
# ------------------------------
def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_torch(model: torch.nn.Module, shp: Shapes, dtype: torch.dtype, iters: int, warmup: int) -> float:
    device = "cuda"
    images, mel, alive = make_inputs(shp, dtype, device=device)
    shim = CS2TRTShim(model.eval().to(device))

    # warmup
    with torch.inference_mode():
        for _ in range(warmup):
            _ = shim(images, mel, alive)
        _sync_cuda()

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = shim(images, mel, alive)
        _sync_cuda()
        t1 = time.perf_counter()

    return (t1 - t0) / iters


def bench_onnx(onnx_path: str, shp: Shapes, dtype: torch.dtype, iters: int, warmup: int) -> Optional[float]:
    if ort is None:
        print("onnxruntime not available; skipping ONNX benchmark.")
        return None

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)

    import numpy as np
    device = "cuda"
    # Create torch tensors then convert to numpy
    images, mel, alive = make_inputs(shp, dtype, device=device)
    images_np = images.detach().cpu().numpy()
    mel_np = mel.detach().cpu().numpy()
    alive_np = alive.detach().cpu().numpy()

    inputs = {
        "images": images_np,
        "mel_spectrogram": mel_np,
        "alive_mask": alive_np
    }

    # warmup
    for _ in range(warmup):
        _ = sess.run(None, inputs)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = sess.run(None, inputs)
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def bench_trt(engine_path: str, shp: Shapes, dtype: torch.dtype, iters: int, warmup: int) -> Optional[float]:
    if trt is None:
        print("tensorrt not available; skipping TRT benchmark.")
        return None
    runner = TRTRunner(engine_path)
    device = "cuda"
    images, mel, alive = make_inputs(shp, dtype, device=device)

    # warmup
    for _ in range(warmup):
        _ = runner(images, mel, alive)
    _sync_cuda()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = runner(images, mel, alive)
    _sync_cuda()
    t1 = time.perf_counter()
    return (t1 - t0) / iters


# ------------------------------
# CLI
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Export model to ONNX / TensorRT and benchmark.")
    p.add_argument("--factory", type=str, required=True,
                   help="Dotted path 'module:callable' that returns an eval()'d CUDA model. Example: model:build_model")
    p.add_argument("--export-onnx", type=str, default=None, help="Path to write ONNX.")
    p.add_argument("--build-trt", type=str, default=None, help="Path to write TensorRT engine (.plan).")
    p.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16", "int8"])
    p.add_argument("--workspace-gb", type=float, default=8.0, help="TensorRT workspace size in GB.")
    p.add_argument("--benchmark", action="store_true", help="Run benchmarks for Torch/ONNX/TRT.")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--opset", type=int, default=18)

    # Shapes
    p.add_argument("--B", type=int, default=1)
    p.add_argument("--T", type=int, default=1)
    p.add_argument("--P", type=int, default=5)
    p.add_argument("--H", type=int, default=480)
    p.add_argument("--W", type=int, default=640)
    p.add_argument("--M", type=int, default=6)

    return p.parse_args()


def main():
    args = parse_args()
    shp = Shapes(args.B, args.T, args.P, args.H, args.W, args.M)
    # dtype
    if args.precision.lower() in ("fp16", "bf16"):
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Instantiate model from factory
    factory = load_factory(args.factory)
    model = factory()  # must return a CUDA, eval() model, compatible with CS2TRTShim

    # Export ONNX
    if args.export_onnx:
        print(f"[ONNX] Exporting to {args.export_onnx} (opset {args.opset}) ...")
        export_to_onnx(model, args.export_onnx, shp, dtype, opset=args.opset)
        print(f"[ONNX] Done → {args.export_onnx}")

    # Build TensorRT
    if args.build_trt:
        if not args.export_onnx and not os.path.exists(args.build_trt):
            print("[TRT] No ONNX given; looking for 'cs2.onnx' in CWD...")
        onnx_path = args.export_onnx or "cs2.onnx"
        print(f"[TRT] Building engine from {onnx_path} to {args.build_trt} (precision={args.precision}) ...")
        build_trt_engine(onnx_path, args.build_trt, args.precision, args.workspace_gb)
        print(f"[TRT] Done → {args.build_trt}")

    # Benchmarks
    if args.benchmark:
        it, wu = args.iters, args.warmup
        print("\n=== Benchmark ===")
        t_torch = bench_torch(model, shp, dtype, it, wu)
        print(f"PyTorch: {t_torch*1000:.2f} ms/iter  |  {1.0/t_torch:.2f} it/s")

        if args.export_onnx and ort is not None:
            t_onnx = bench_onnx(args.export_onnx, shp, dtype, it, wu)
            if t_onnx is not None:
                print(f"ONNX RT: {t_onnx*1000:.2f} ms/iter  |  {1.0/t_onnx:.2f} it/s")
        else:
            print("ONNX RT: (skipped) export --export-onnx first and ensure onnxruntime is installed.")

        if args.build_trt and trt is not None:
            t_trt = bench_trt(args.build_trt, shp, dtype, it, wu)
            if t_trt is not None:
                print(f"TensorRT: {t_trt*1000:.2f} ms/iter  |  {1.0/t_trt:.2f} it/s")
        else:
            print("TensorRT: (skipped) build with --build-trt and ensure TensorRT is installed.")


if __name__ == "__main__":
    main()
