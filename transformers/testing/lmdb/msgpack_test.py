import timeit
import numpy as np
import msgpack
import pandas as pd

# 1. Prepare four 100-dimensional numpy vectors
vectors = [np.random.rand(100).tolist() for _ in range(4)]

# 2. Define packing function using msgpack
def pack_msgpack():
    msgpack.packb(vectors, use_bin_type=True)

# 3. Benchmark settings
number = 200

# 4. Run benchmark
t_msgpack = timeit.timeit('pack_msgpack()', globals=globals(), number=number)

# 5. Prepare results
results = pd.DataFrame({
    'Method': ['MessagePack packing'],
    'Loops': [number],
    'Total Time (s)': [t_msgpack],
    'Ops per Second': [number / t_msgpack]
})
print(results)