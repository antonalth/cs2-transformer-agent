# bench_dali_video.py
# Benchmark DALI video decoding on GPU and verify padding + frame selection.
# Usage:
#   python bench_dali_video.py --file sample.mp4 --start 500 --end 1000 --batch-size 4 --iters 200 --size 224 224

import argparse, os, tempfile, time
import numpy as np

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn

def make_file_list(path: str, start: int, end: int) -> str:
    """Create a temp DALI file_list with one line: <abs_path>  <label>  <start_frame>  <end_frame>."""
    abs_path = os.path.abspath(path)
    text = f"{abs_path} 0 {start} {end}\n"
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w")
    tf.write(text)
    tf.close()
    return tf.name

@pipeline_def
def dali_video_pipe(file_list, seq_len, resize_hw=None):
    # Use readers.video so enable_frame_num yields frame indices as a 3rd output
    video, labels, frame_nums = fn.readers.video(
        device="gpu",
        file_list=file_list,
        file_list_frame_num=True,             # interpret start/end as frame indices
        sequence_length=seq_len,
        step=seq_len,                         # one clip per line
        random_shuffle=False,
        enable_frame_num=True,                # return per-frame indices
        pad_sequences=True,                   # zero-pad beyond EOF; frame_nums = -1 for padded frames
        file_list_include_preceding_frame=False,  # silence the warning; make behavior explicit
        name="video_reader",
    )
    if resize_hw is not None:
        h, w = resize_hw
        video = fn.resize(video, resize_y=h, resize_x=w)
    return video, labels, frame_nums  # [B,F,H,W,C], [B], [B,F]

def _run_and_unpack(pipe):
    """Run the pipeline and always return (video_tl, labels_tl, frame_nums_tl or None)."""
    outs = pipe.run()
    video_tl = outs[0]
    labels_tl = outs[1] if len(outs) > 1 else None
    frame_nums_tl = outs[2] if len(outs) > 2 else None
    return video_tl, labels_tl, frame_nums_tl

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
        video_tl, labels_tl, frame_nums_tl = _run_and_unpack(pipe)
        _ = video_tl.as_cpu().as_array()  # host copy only for warmup/verify

    # Timed loop
    t0 = time.time()
    for _ in range(args.iters):
        _run_and_unpack(pipe)  # decode stays on GPU
    t1 = time.time()
    elapsed = t1 - t0
    total_frames = args.batch_size * seq_len * args.iters
    fps = total_frames / max(elapsed, 1e-9)

    # One verification fetch (host copy for printing / padding checks)
    video_tl, labels_tl, frame_nums_tl = _run_and_unpack(pipe)
    video = video_tl.as_cpu().as_array()                   # [B,F,H,W,C] uint8
    frame_nums = (frame_nums_tl.as_cpu().as_array()
                  if frame_nums_tl is not None else None) # [B,F] int32

    print("\n=== DALI decode summary ===")
    print(f"file:         {os.path.abspath(args.file)}")
    print(f"range:        [{args.start}, {args.end})  (seq_len={seq_len})")
    print(f"batch size:   {args.batch_size}")
    print(f"iters:        {args.iters} (warmup={args.warmup})")
    print(f"throughput:   {fps:,.1f} frames/sec  ({elapsed:.3f}s total)")
    print(f"video shape:  {video.shape}  dtype={video.dtype}  (layout: [B,F,H,W,C])")

    # Show first sample’s frame indices and detect padding
    if frame_nums is not None:
        fnums0 = frame_nums[0]                                # [F]
        pad_mask = (fnums0 == -1)
        num_pad = int(pad_mask.sum())
        head = fnums0[:min(8, fnums0.shape[0])].tolist()
        tail = fnums0[max(0, fnums0.shape[0]-8):].tolist()
        print("\nFrame indices (sample 0):")
        print(f"  head: {head}")
        print(f"  tail: {tail}")
        if num_pad > 0:
            print(f"  padding detected: {num_pad} trailing frame(s) with index -1 "
                  "(zero-filled pixels per pad_sequences).")
            zeros_ok = bool((video[0, pad_mask] == 0).all()) if video.shape[1] == fnums0.shape[0] else True
            print(f"  padded frames are zero: {zeros_ok}")
        else:
            print("  no padding in this window.")
        valid = fnums0[fnums0 >= 0]
        if valid.size > 0:
            start_ok = int(valid[0]) == args.start
            step_ok = bool(np.all(np.diff(valid) == 1))
            print("\nSelection check:")
            print(f"  starts at requested start: {start_ok} (got {int(valid[0])})")
            print(f"  consecutive frames step=1: {step_ok}")
    else:
        # Fallback if your DALI build doesn’t provide frame numbers:
        tail_zero = bool((video[0, -8:] == 0).all())
        print("\nFrame indices not provided by reader; "
              f"zero-tail padding detected: {tail_zero}")

    # Clean up temp file_list
    try:
        os.remove(file_list)
    except OSError:
        pass

if __name__ == "__main__":
    main()
