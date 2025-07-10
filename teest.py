import geopack
from geopack import t89, t96

# Time setup
ut = 100  # 1970-01-01/00:01:40 UT
ps = geopack.recalc(ut)

# Calculate field at a single point using original scalar function
x, y, z = 5.0, 0.0, 0.0
bx, by, bz = t89(3, ps, x, y, z)  # Kp = 3
print(f"T89 field: Bx={bx:.2f}, By={by:.2f}, Bz={bz:.2f} nT")

# T96 with solar wind parameters
parmod = [2.0, -20.0, 0.0, -5.0, 0, 0, 0, 0, 0, 0]  # [Pdyn, Dst, ByIMF, BzIMF, ...]
bx, by, bz = t96(parmod, ps, x, y, z)
from geopack.vectorized import t89_vectorized, t96_vectorized
import numpy as np

# OPTION 1: Single point (works just like scalar version)
x, y, z = 5.0, 0.0, 0.0
bx, by, bz = t89_vectorized(3, ps, x, y, z)  # Returns scalars
print(f"T89 vectorized (single): Bx={bx:.2f}, By={by:.2f}, Bz={bz:.2f} nT")

# OPTION 2: Multiple points at once (this is where vectorization shines!)
x_array = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
y_array = np.zeros(5)
z_array = np.zeros(5)
bx_array, by_array, bz_array = t89_vectorized(3, ps, x_array, y_array, z_array)
print(f"Shape of output: {bx_array.shape}")  # (5,)
print(f"First point: Bx={bx_array[0]:.2f} nT")

# OPTION 3: Large arrays for maximum performance
x_large = np.linspace(-10, 10, 10000)
y_large = np.zeros(10000)
z_large = np.zeros(10000)
bx_large, by_large, bz_large = t89_vectorized(3, ps, x_large, y_large, z_large)
# Processes all 10,000 points in ~0.1 seconds!
import time

# Generate test data
n_points = 1000
x = np.random.uniform(-10, 5, n_points)
y = np.random.uniform(-5, 5, n_points)
z = np.random.uniform(-3, 3, n_points)

# Scalar approach (original geopack)
start = time.time()
bx_scalar = np.zeros(n_points)
by_scalar = np.zeros(n_points)
bz_scalar = np.zeros(n_points)
for i in range(n_points):
    bx_scalar[i], by_scalar[i], bz_scalar[i] = t89(3, ps, x[i], y[i], z[i])
scalar_time = time.time() - start

# Vectorized approach (new)
start = time.time()
bx_vec, by_vec, bz_vec = t89_vectorized(3, ps, x, y, z)
vector_time = time.time() - start

print(f"Scalar time: {scalar_time:.3f} s")
print(f"Vectorized time: {vector_time:.3f} s")
print(f"Speedup: {scalar_time/vector_time:.0f}x")
print(f"Results match: {np.allclose(bx_scalar, bx_vec)}")  # True