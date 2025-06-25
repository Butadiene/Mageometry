import numpy as np
from geopack.t96 import birk1tot_02, condip1

# Test point
x, y, z = 5.0, 3.0, 1.0
ps = 0.0

# Initialize globals
_ = birk1tot_02(ps, x, y, z)

# Get scalar condip1 result
xi = [x, y, z, ps]
d_scalar = condip1(xi)

print("Scalar d array (first 5 values - conical harmonics):")
for m in range(5):
    print(f"  d[0,{m}] = {d_scalar[0,m]:.8f}")  # Bx component
    print(f"  d[1,{m}] = {d_scalar[1,m]:.8f}")  # By component
    print(f"  d[2,{m}] = {d_scalar[2,m]:.8f}")  # Bz component
    print()

# Model coefficients
c2 = np.array([
    6.04133, .305415, .606066e-02, .128379e-03, -.179406e-04
])

print("\nContributions with coefficients:")
for m in range(5):
    bx_contrib = c2[m] * d_scalar[0,m]
    by_contrib = c2[m] * d_scalar[1,m]
    bz_contrib = c2[m] * d_scalar[2,m]
    print(f"  m={m}: Bx={bx_contrib:.8f}, By={by_contrib:.8f}, Bz={bz_contrib:.8f}")