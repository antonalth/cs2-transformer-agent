# bench_multi_pov_5.py
# Decode+resize 5 concurrent POV streams with DALI and report throughput.
# Requires an NVIDIA driver with NVDEC (libnvcuvid), and a DALI wheel matching your CUDA.

import argparse, os, tempfile, time
from typing import List, Tuple
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn

def make_file_list(path: str, start: int, end: int, lines: int) -> str:
    """
    Create a temp DALI file_list with `lines` identical entries:
      <abs_path>  <label>  <start_frame>  <end_frame>
    start/end are treated as frame indices by the reader.
    """
    abs_path = os.path.abspath(path)
    text = "\n".join(f"{abs_path} 0 {start} {end}" for _ in range(lines)) + "\n"
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w")
    tf.write(text)
    tf.close()
    return tf.name

def make_5_file_lists(files5: List[str], start: int, end: int, batch_size: int) -> List[str]:
    """Return a list of 5 file_list paths, one per POV."""
    return [make_file_list(files5[i], start, end, batch_size) for i in range(5)]

@pipeline_def
def pipe_5pov(file_lists: List[str], seq_len: int, H: int, W: int,
              add_surfaces: int, read_ahead: bool):
    """
    Build 5 parallel video readers (decode+resize fused) and return only the video outputs.
    Each output is a uint8 tensor of shape [B, F, H, W, C] on GPU (CUDA).
    """
    outs = []
    for i in range(5):
        # readers.video_resize returns (video, labels); we keep only `video`.
        v, _ = fn.readers.video_resize(
            device="gpu",
            file_list=file_lists[i],
            file_list_frame_num=True,            # interpret start/end as frame indices
            sequence_length=seq_len, step=seq_len,  # one T-length clip per line
            random_shuffle=False,
            pad_sequences=True,                  # zero-pad tail past EOF
            read_ahead=read_ahead,
            additional_decode_surfaces=add_surfaces,
            resize_y=H, resize_x=W,
            name=f"vid{i}",
        )
        outs.append(v)  # uint8 [B, F, H, W, C]
    return tuple(outs)

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--files", nargs=5, help="Five MP4s: p0 p1 p2 p3 p4")
    g.add_argument("--file", help="Single MP4 to reuse for all 5 readers (for testing)")
    ap.add_argument("--start", type=int, default=500)
    ap.add_argument("--end", type=int, default=1000, help="exclusive; seq_len=end-start")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--size", type=int, nargs=2, default=[224, 224], metavar=("H","W"))
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--prefetch", type=int, default=4)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--add-surfaces", type=int, default=12, help="additional_decode_surfaces")
    ap.add_argument("--read-ahead", action="store_true", help="enable read_ahead in the reader")
    args = ap.parse_args()

    if args.files:
        if len(args.files) != 5:
            ap.error("--files requires exactly 5 paths")
        files5 = args.files
    else:
        files5 = [args.file] * 5

    assert args.end > args.start, "end must be > start"
    seq_len = args.end - args.start
    H, W = args.size

    # Build per-POV file_lists (one line per batch sample)
    file_lists = make_5_file_lists(files5, args.start, args.end, args.batch_size)

    # Construct pipeline
    pipe = pipe_5pov(
        batch_size=args.batch_size,
        num_threads=args.threads,
        device_id=args.device_id,
        prefetch_queue_depth=args.prefetch,
        file_lists=file_lists,
        seq_len=seq_len,
        H=H, W=W,
        add_surfaces=args.add_surfaces,
        read_ahead=args.read_ahead,
    )
    pipe.build()

    # Warmup
    for _ in range(args.warmup):
        _ = pipe.run()

    # Timed loop
    t0 = time.time()
    for _ in range(args.iters):
        _ = pipe.run()   # returns 5 TensorLists: one per POV
    t1 = time.time()

    # Compute throughput (frames/sec) across all 5 POVs
    total_frames = args.batch_size * seq_len * args.iters * 5
    fps = total_frames / max(t1 - t0, 1e-9)

    # Fetch one batch to report shapes
    outs = pipe.run()  # tuple of 5 TensorLists
    shapes = []
    for i, tl in enumerate(outs):
        arr = tl.as_cpu().as_array()  # [B, F, H, W, C] uint8
        shapes.append((i, arr.shape, arr.dtype))

    print("\n=== DALI 5-POV decode+resize benchmark ===")
    print(f"files:       {[os.path.abspath(p) for p in files5]}")
    print(f"range:       [{args.start}, {args.end})  (seq_len={seq_len})")
    print(f"batch size:  {args.batch_size}   iters: {args.iters}   warmup: {args.warmup}")
    print(f"size:        {H}x{W}   threads: {args.threads}   prefetch: {args.prefetch}")
    print(f"read_ahead:  {bool(args.read_ahead)}   add_surfaces: {args.add_surfaces}")
    print(f"throughput:  {fps:,.1f} frames/sec  (across all 5 POVs)")
    for i, shp, dt in shapes:
        print(f"p{i} shape:  {shp}  dtype={dt}  (layout: [B,F,H,W,C])")

    # Cleanup temp file_lists
    for fl in file_lists:
        try: os.remove(fl)
        except OSError: pass

if __name__ == "__main__":
    main()
