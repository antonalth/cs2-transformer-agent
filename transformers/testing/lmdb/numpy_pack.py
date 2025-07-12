import timeit
import numpy as np

# Define structured dtype: 8 ints, 8 floats, 400 bits (packed into 50 bytes)
dtype = np.dtype([
    ('ints',   np.int32,   8),
    ('floats', np.float32, 8),
    ('bits',   np.uint8,   50),
])

# Prepare a single sample record
record = np.zeros((), dtype=dtype)
record['ints']   = np.arange(8, dtype=np.int32)
record['floats'] = np.linspace(0, 1, 8, dtype=np.float32)
# Create a random mask of length 400, pack into 50 bytes
bools = np.random.randint(0, 2, size=400, dtype=np.uint8)
record['bits'] = np.packbits(bools, bitorder='big')

# Functions to pack and unpack
def pack_numpy():
    _ = record.tobytes()

def unpack_numpy():
    blob = record.tobytes()
    rec_back = np.frombuffer(blob, dtype=dtype)[0]
    _ = rec_back['ints'], rec_back['floats'], rec_back['bits']

# Benchmark settings
iterations = 20000

# Run benchmarks
pack_time   = timeit.timeit('pack_numpy()', globals=globals(), number=iterations)
unpack_time = timeit.timeit('unpack_numpy()', globals=globals(), number=iterations)

print(f"NumPy packing   : {iterations} runs in {pack_time:.6f}s → {iterations/pack_time:.0f} ops/sec")
print(f"NumPy unpacking : {iterations} runs in {unpack_time:.6f}s → {iterations/unpack_time:.0f} ops/sec")
