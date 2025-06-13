import time
import struct

# Benchmark parameters
ITERATIONS = 100_000
AVG_JPEG_SIZE = 100_000   # simulate 100 KB JPEG blobs
INPUT_SIZE = 16           # simulate 16 bytes of input bitfield

# Metadata struct: example fields (keys_bitmask, d_yaw, d_pitch)
# uint64 + float32 + float32 = 8 + 4 + 4 = 16 bytes
META_FMT = ">Qff"
struct_packer = struct.Struct(META_FMT)
struct_unpacker = struct.Struct(META_FMT)

# Simulate data
meta_tuple = (1234567890123456789, 0.1234, -0.5678)
meta_blob = struct_packer.pack(*meta_tuple)  # 16-byte header
jpeg_blob = b'\xff' * AVG_JPEG_SIZE         # 100 KB dummy JPEG data
input_blob = b'\xAA' * INPUT_SIZE           # 16-byte input data

# Pack function
def pack_record(meta: bytes, jpeg: bytes, inp: bytes) -> bytes:
    return meta + jpeg + inp

# Unpack function
def unpack_record(blob: bytes):
    mv = memoryview(blob)
    meta_vals = struct_unpacker.unpack(mv[:struct_unpacker.size])
    # Slice out JPEG and input without copying until needed
    jpeg = mv[struct_unpacker.size : struct_unpacker.size + AVG_JPEG_SIZE]
    inp  = mv[struct_unpacker.size + AVG_JPEG_SIZE : ]
    return meta_vals, jpeg, inp

# Warm-up to account for any initial overhead
packed = pack_record(meta_blob, jpeg_blob, input_blob)
_ = unpack_record(packed)

# Benchmark packing
start_pack = time.perf_counter()
for _ in range(ITERATIONS):
    packed = pack_record(meta_blob, jpeg_blob, input_blob)
end_pack = time.perf_counter()

# Benchmark unpacking
start_unpack = time.perf_counter()
for _ in range(ITERATIONS):
    meta, jpeg, inp = unpack_record(packed)
end_unpack = time.perf_counter()

# Calculate timings
total_pack_time = end_pack - start_pack
total_unpack_time = end_unpack - start_unpack
avg_pack_us   = (total_pack_time   / ITERATIONS) * 1e6  # µs
avg_unpack_us = (total_unpack_time / ITERATIONS) * 1e6  # µs
combined_us   = avg_pack_us + avg_unpack_us
fps           = 1.0 / (combined_us * 1e-6)

# Results
print(f"Pack   avg time: {avg_pack_us:.3f} µs over {ITERATIONS} iterations")
print(f"Unpack avg time: {avg_unpack_us:.3f} µs over {ITERATIONS} iterations")
print(f"Combined avg:   {combined_us:.3f} µs  (~{fps:.0f} records/sec)")
