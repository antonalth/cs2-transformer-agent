import timeit
import bitstruct

# 1. Compile format
fmt  = bitstruct.compile('>' + 'u1' * 400)
# 2. Build args tuple once (400 ints)
args = tuple([1] * 400)

# 3. Benchmark pure pack
number = 500_000
stmt  = 'fmt.pack(*args)'
setup = 'from __main__ import fmt, args'
t     = timeit.timeit(stmt, setup, number=number)

print(f"{number} pack calls in {t:.3f}s → {number/t:.0f} ops/sec")
