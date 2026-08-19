# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import time
import pandas as pd
from datetime import datetime

# Import geopack modules
from mageometry import geopack
from mageometry.geopack import t96, t96_vectorized

# Set up time and recalculate parameters
ut = datetime(2015, 3, 17, 12, 0, 0).timestamp()
ps = geopack.recalc(ut)

# T96 model parameters: [Pdyn, Dst, ByIMF, BzIMF, unused...]
parmod = [2.0,  # Solar wind dynamic pressure (nPa)
          -50,  # Dst index (nT)
          0.0,  # IMF By (nT)
          -5.0, # IMF Bz (nT)
          0, 0, 0, 0, 0, 0]  # Unused parameters

# Set up plotting

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


# Create 2D grid in X-Z plane
x = np.linspace(-15, 10, 100)
z = np.linspace(-10, 10, 100)
X, Z = np.meshgrid(x, z)
Y = np.zeros_like(X)

# Flatten for vectorized calculation
x_flat = X.flatten()
y_flat = Y.flatten()
z_flat = Z.flatten()

# Calculate field using scalar T96 (loop over all points)
print("=" * 60)
print("Calculating field using SCALAR T96...")
print(f"Grid size: {len(x_flat)} points")
start_time = time.time()
bx_scalar = np.zeros_like(x_flat)
by_scalar = np.zeros_like(y_flat)
bz_scalar = np.zeros_like(z_flat)
for i in range(len(x_flat)):
    bx_scalar[i], by_scalar[i], bz_scalar[i] = t96(parmod, ps, x_flat[i], y_flat[i], z_flat[i])
scalar_time = time.time() - start_time
print(f"Scalar T96 completed in {scalar_time:.3f} seconds")
print(f"Processing rate: {len(x_flat)/scalar_time:.0f} points/second")

# Calculate field using vectorized T96
print("\n" + "=" * 60)
print("Calculating field using VECTORIZED T96...")
start_time = time.time()
bx_flat, by_flat, bz_flat = t96_vectorized(parmod, ps, x_flat, y_flat, z_flat)
vectorized_time = time.time() - start_time
print(f"Vectorized T96 completed in {vectorized_time:.3f} seconds")
print(f"Processing rate: {len(x_flat)/vectorized_time:.0f} points/second")

# Compare results
print("\n" + "=" * 60)
print("PERFORMANCE COMPARISON:")
print(f"Speedup: {scalar_time/vectorized_time:.1f}x faster")
print(f"Time saved: {scalar_time - vectorized_time:.2f} seconds")

# Verify accuracy
max_diff = np.max([
    np.abs(bx_scalar - bx_flat).max(),
    np.abs(by_scalar - by_flat).max(),
    np.abs(bz_scalar - bz_flat).max()
])
print(f"\nAccuracy check - Max absolute difference: {max_diff:.2e} nT")
print("=" * 60)

# Reshape results
Bx = bx_flat.reshape(X.shape)
By = by_flat.reshape(X.shape)
Bz = bz_flat.reshape(X.shape)
B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)

# Calculate distance from Earth's center
R = np.sqrt(X**2 + Y**2 + Z**2)

# Mask field magnitude for regions closer than 1 Re
B_mag_masked = np.where(R >= 1.0, B_mag, np.nan)

# Log magnetic field strength statistics
print(f"\nMagnetic Field Strength Statistics:")
print(f"  Min |B|: {B_mag.min():.2f} nT")
print(f"  Max |B|: {B_mag.max():.2f} nT")
print(f"  Mean |B|: {B_mag.mean():.2f} nT")
print(f"  Std |B|: {B_mag.std():.2f} nT")

# Plot field magnitude with logarithmic scale for better visibility
plt.figure(figsize=(12, 8))

# Use masked data for plotting (NaN values won't be colored)
# Set minimum value to avoid log(0), using nanmin/nanmax to ignore NaN
B_mag_plot = B_mag_masked
B_min = np.nanmin(B_mag_plot)
B_max = np.nanmax(B_mag_plot)

# Create custom levels for better distribution
levels = np.logspace(np.log10(B_min), np.log10(B_max), 50)

# Use a perceptually uniform colormap with better contrast
contour = plt.contourf(X, Z, B_mag_plot, levels=levels, cmap='plasma', 
                       norm=LogNorm(vmin=B_min, vmax=B_max))
cbar = plt.colorbar(contour, label='|B| (nT, r ≥ 1 Re)', format='%.0f')

# Add circle to show Earth boundary at 1 Re
circle = plt.Circle((0, 0), 1.0, fill=True, color='white', zorder=10)
plt.gca().add_patch(circle)
plt.text(0, 0, 'Earth\n(1 Re)', ha='center', va='center', fontsize=10, zorder=11)

# Log color map information
print(f"\nColor Map Information:")
print(f"  Colormap: plasma (perceptually uniform)")
print(f"  Scale: Logarithmic")
print(f"  Number of levels: 50")
print(f"  Color range: {B_min:.2f} - {B_max:.2f} nT (r ≥ 1 Re only)")
print(f"  Regions inside 1 Re are shown in white")

# Calculate vector magnitudes for quiver plot (normalized in X-Z plane)
vector_mag = np.sqrt(Bx**2 + Bz**2) / B_mag

# Add field lines with corrected vector magnitude
skip = 5
plt.quiver(X[::skip, ::skip], Z[::skip, ::skip], 
           Bx[::skip, ::skip]/B_mag[::skip, ::skip], 
           Bz[::skip, ::skip]/B_mag[::skip, ::skip],
           vector_mag[::skip, ::skip],
           alpha=0.5, scale=30, cmap='plasma')

plt.xlabel('X (Re)')
plt.ylabel('Z (Re)')
plt.title(f'T96 Magnetic Field in X-Z Plane (Y=0)\n(Vectorized: {vectorized_time:.2f}s vs Scalar: {scalar_time:.2f}s - {scalar_time/vectorized_time:.1f}x speedup)')
plt.axis('equal')
plt.show()