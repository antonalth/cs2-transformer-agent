#!/usr/bin/env python3
import argparse, time, torch
from transformers import AutoModel

def build_hf_dinov3():
    core = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
    class Wrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m.eval()
        def forward(self, x):                    # x: [B,3,H,W], float
            out = self.m(pixel_values=x)         # returns BaseModelOutput
            return out.last_hidden_state         # [B, N_tokens, C], tensor only for compile()
    return Wrapper(core)

def main():
    ap = argparse.ArgumentParser()
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

    # --- Perf knobs: Flash/SDPA, TF32, matmul ---
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)

    # --- Model ---
    assert args.height % 16 == 0 and args.width % 16 == 0, "Use multiples of 16 for ViT/16."
    model = build_hf_dinov3().to(device)

    # Optionally cast model params for memory/speed
    if args.param_dtype == "fp16":
        model = model.half()
    elif args.param_dtype == "bf16":
        model = model.to(dtype=torch.bfloat16)

    model = torch.compile(model, mode=args.compile_mode, fullgraph=False, dynamic=False)

    # --- AMP dtype for compute ---
    if args.amp == "fp16":
        amp_dtype = torch.float16
    elif args.amp == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float32

    B, C, H, W = args.batch_size, 3, args.height, args.width
    tokens = (H // 16) * (W // 16)

    # --- Dummy input ---
    x = torch.randn(B, C, H, W, device=device)
    x = x.to(memory_format=torch.channels_last)  # better tensor cores utilization

    # --- Warmup ---
    torch.cuda.synchronize()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype!=torch.float32)):
        for _ in range(args.warmup):
            _ = model(x)
    torch.cuda.synchronize()

    # --- Benchmark ---
    torch.cuda.reset_peak_memory_stats()
    times = []
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
    print(f"Model          : facebook/dinov3-vitb16-pretrain-lvd1689m (HF)")
    print(f"Input          : {B}x3x{H}x{W}  (patches: {H//16}x{W//16}={tokens})")
    print(f"AMP compute    : {args.amp}")
    print(f"Param dtype    : {args.param-dtype if False else args.param_dtype}")  # keep print friendly
    print(f"torch.compile  : mode={args.compile_mode}")
    print(f"SDPA kernels   : flash=True, mem_efficient=True, math=False")
    print(f"Iters/Warmup   : {args.iters}/{args.warmup}")
    print(f"Avg latency    : {avg*1000:.2f} ms/iter")
    print(f"P50 latency    : {p50*1000:.2f} ms/iter")
    print(f"Throughput     : {fps:.2f} img/s")
    print(f"Peak VRAM      : {peak_mem:.1f} MB")

if __name__ == "__main__":
    main()
