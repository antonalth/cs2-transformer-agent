#!/usr/bin/env python3
import argparse, time, torch
from transformers import AutoModel
from contextlib import nullcontext

# Prefer the new SDPA API; fall back gracefully if not available
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    def sdpa_pref():
        # Prefer Flash, then mem-efficient, avoid math kernel
        return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    SDPA_DESC = "flash=True, efficient=True"
except Exception:
    def sdpa_pref():
        # Older PyTorch: do nothing (or you could enable the deprecated API)
        return nullcontext()
    SDPA_DESC = "default (PyTorch chooses)"

def build_hf_dinov2(model_id="facebook/dinov2-base"):
    core = AutoModel.from_pretrained(model_id)
    class Wrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m.eval()
        def forward(self, x):                     # x: [B,3,H,W], float
            out = self.m(pixel_values=x)          # BaseModelOutput
            return out.last_hidden_state          # [B, N_tokens, C]
    return Wrapper(core)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="facebook/dinov2-base",
                    help="HF model id, e.g. facebook/dinov2-base|large|giant")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--amp", choices=["fp16","bf16","fp32"], default="fp16")
    ap.add_argument("--param-dtype", choices=["fp32","fp16","bf16"], default="fp32",
                    help="Optional cast of model weights for memory/speed.")
    ap.add_argument("--compile-mode", default="max-autotune")   # or "reduce-overhead"
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA is required."
    device = "cuda"

    # --- Perf knobs: TF32-friendly GEMMs ---
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # --- Model ---
    model = build_hf_dinov2(args.model).to(device)

    # Infer patch size from config (DINOv2 ViTs use 14)
    patch_size = getattr(getattr(model, "m", model), "config", None)
    patch_size = getattr(patch_size, "patch_size", 14)

    # Optionally cast params for memory/speed
    if args.param_dtype == "fp16":
        model = model.half()
    elif args.param_dtype == "bf16":
        model = model.to(dtype=torch.bfloat16)

    model = torch.compile(model, mode=args.compile_mode, fullgraph=False, dynamic=False)

    # --- AMP compute dtype ---
    if args.amp == "fp16":
        amp_dtype = torch.float16
    elif args.amp == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float32

    # Effective H/W: crop down to nearest multiple of patch size (DINOv2 drops remainder)
    req_H, req_W = args.height, args.width
    H = req_H - (req_H % patch_size)
    W = req_W - (req_W % patch_size)
    B, C = args.batch_size, 3
    grid_h, grid_w = H // patch_size, W // patch_size
    tokens = grid_h * grid_w

    # --- Dummy input ---
    x = torch.randn(B, C, H, W, device=device)
    x = x.to(memory_format=torch.channels_last)

    # --- Warmup ---
    torch.cuda.synchronize()
    with sdpa_pref():
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype!=torch.float32)):
            for _ in range(args.warmup):
                _ = model(x)
    torch.cuda.synchronize()

    # --- Benchmark ---
    torch.cuda.reset_peak_memory_stats()
    times = []
    with sdpa_pref():
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype!=torch.float32)):
            for _ in range(args.iters):
                t0 = time.perf_counter()
                _ = model(x)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append(t1 - t0)

    total = sum(times)
    avg = total / len(times)
    p50 = sorted(times)[len(times)//2]
    fps = (args.iters * B) / total
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)

    # --- Info ---
    cc = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    print(f"Device         : {name} (cc {cc[0]}.{cc[1]}) | CUDA {torch.version.cuda} | torch {torch.__version__}")
    print(f"Model          : {args.model} (HF, ViT patch={patch_size})")
    if (H, W) != (req_H, req_W):
        print(f"Requested HW   : {req_H}x{req_W}  ->  Effective HW: {H}x{W} (cropped to /{patch_size})")
    else:
        print(f"Input          : {B}x3x{H}x{W}")
    print(f"Patches        : {grid_h}x{grid_w} = {tokens} tokens")
    print(f"AMP compute    : {args.amp}")
    print(f"Param dtype    : {args.param_dtype}")
    print(f"torch.compile  : mode={args.compile_mode}")
    print(f"SDPA kernels   : {SDPA_DESC}")
    print(f"Iters/Warmup   : {args.iters}/{args.warmup}")
    print(f"Avg latency    : {avg*1000:.2f} ms/iter")
    print(f"P50 latency    : {p50*1000:.2f} ms/iter")
    print(f"Throughput     : {fps:.2f} img/s")
    print(f"Peak VRAM      : {peak_mem:.1f} MB")

if __name__ == "__main__":
    main()
