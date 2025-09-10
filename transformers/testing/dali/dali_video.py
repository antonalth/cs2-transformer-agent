# bench_dali_video.py
# Benchmark DALI video decoding on GPU and verify padding + frame selection.
# Requires: pip install nvidia-dali-cuda*  (pick the build matching your CUDA)
# Docs: readers.video/readers.video_resize, pad_sequences, enable_frame_num.  # noqa

import argparse, os, tempfile, time
import numpy as np

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

def make_file_list(path: str, start: int, end: int) -> str:
    """
    Create a temporary DALI file_list with a single sample:
      <abs_path>  <label>  <start_frame>  <end_frame>
    Start/end are interpreted as frame numbers with file_list_frame_num=True.
    """
    abs_path = os.path.abspath(path)
    text = f"{abs_path} 0 {start} {end}\n"
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w")
    tf.write(text)
    tf.close()
    return tf.name

@pipeline_def
def dali_video_pipe(file_list, seq_len, resize_hw=None):
    """
    When resize_hw is provided, uses readers.video_resize (decode+resize in one op).
    We enable frame numbers so we can detect any padding (=-1) at the tail.
    """
    common = dict(
        device="gpu",
        file_list=file_list,
        file_list_frame_num=True,   # treat start/end as frame indices
        sequence_length=seq_len,
        step=seq_len,               # one clip per line
        random_shuffle=False,
        enable_frame_num=True,      # get per-frame indices to verify/padding
        pad_sequences=True,         # zero-pad past EOF if needed
        name="video_reader",
    )
    if resize_hw is None:
        video, labels, frame_nums = fn.readers.video(**common)
    else:
        h, w = resize_hw
        video, labels, frame_nums = fn.readers.video_resize(resize_y=h, resize_x=w, **common)
    # Keep raw uint8 to make zero-padding easy to check; layout: [F,H,W,C]
    return video, frame_nums

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, default="sample.mp4")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=64, help="exclusive; seq_len=end-start")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--iters", type=int, default=200, help="timed iterations (after warmup)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--size", type=int, nargs=2, default=None, metavar=("H","W"),
                    help="optional resize (e.g., 224 224). If omitted, native size is used.")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    assert args.end > args.start, "end must be > start"
    seq_len = args.end - args.start

    file_list = make_file_list(args.file, args.start, args.end)

    pipe = dali_video_pipe(
        batch_size=args.batch_size,
        num_threads=args.threads,
        device_id=args.device_id,
        file_list=file_list,
        seq_len=seq_len,
        resize_hw=tuple(args.size) if args.size else None,
        prefetch_queue_depth=2,
    )
    pipe.build()

    # Warmup
    for _ in range(args.warmup):
        video_tl, labels_tl, frame_nums_tl = pipe.run()
        # .as_cpu().as_array() for host-side inspection (only during warmup/verify)
        _ = video_tl.as_cpu().as_array()

    # Timed loop
    t0 = time.time()
    for _ in range(args.iters):
        pipe.run()   # decode stays on GPU; host sync implied at the op boundary
    t1 = time.time()
    elapsed = t1 - t0
    total_frames = args.batch_size * seq_len * args.iters
    fps = total_frames / max(elapsed, 1e-9)

    # One verification fetch (host copy just for printing / padding checks)
    video_tl, frame_nums_tl = pipe.run()
    video = video_tl.as_cpu().as_array()          # [B, F, H, W, C] uint8
    frame_nums = frame_nums_tl.as_cpu().as_array()# [B, F] int32 (padded indices are -1)

    B, F = frame_nums.shape[0], frame_nums.shape[1]
    print("\n=== DALI decode summary ===")
    print(f"file:         {os.path.abspath(args.file)}")
    print(f"range:        [{args.start}, {args.end})  (seq_len={seq_len})")
    print(f"batch size:   {args.batch_size}")
    print(f"iters:        {args.iters} (warmup={args.warmup})")
    print(f"throughput:   {fps:,.1f} frames/sec  ({elapsed:.3f}s total)")
    print(f"video shape:  {video.shape}  dtype={video.dtype}  (layout: [B,F,H,W,C])")

    # Show first sample’s frame indices (head/tail) and detect padding
    fnums0 = frame_nums[0]                        # shape [F]
    pad_mask = (fnums0 == -1)
    num_pad = int(pad_mask.sum())
    head = fnums0[:min(8, F)].tolist()
    tail = fnums0[max(0, F-8):].tolist()
    print("\nFrame indices (sample 0):")
    print(f"  head: {head}")
    print(f"  tail: {tail}")
    if num_pad > 0:
        print(f"  padding detected: {num_pad} trailing frame(s) with index -1 "
              "(zero-filled pixels per DALI pad_sequences).")
        # Sanity: check that those frames are really all zeros
        zeros_ok = bool((video[0, pad_mask] == 0).all()) if video.shape[1] == F else True
        print(f"  padded frames are zero: {zeros_ok}")
    else:
        print("  no padding in this window.")

    # Quick selection sanity: non-padded frames should start at `start` and step by 1
    valid = fnums0[fnums0 >= 0]
    if valid.size > 0:
        start_ok = int(valid[0]) == args.start
        step_ok = bool(np.all(np.diff(valid) == 1))
        print(f"\nSelection check:")
        print(f"  starts at requested start: {start_ok} (got {int(valid[0])})")
        print(f"  consecutive frames step=1: {step_ok}")
    else:
        print("\nSelection check: no valid frames returned (all padded?)")

    # Clean up temp file_list
    try:
        os.remove(file_list)
    except OSError:
        pass

if __name__ == "__main__":
    main()
