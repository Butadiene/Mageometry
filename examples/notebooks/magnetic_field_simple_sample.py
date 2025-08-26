# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import time
import pandas as pd
from datetime import datetime

# Import geopack modules
import geopack
from geopack import t96, t96_vectorized

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

# Calculate field
print("Calculating field on 10,000 point grid...")
start_time = time.time()
bx_flat, by_flat, bz_flat = t96_vectorized(parmod, ps, x_flat, y_flat, z_flat)
calc_time = time.time() - start_time
print(f"Calculation completed in {calc_time:.3f} seconds")
print(f"Processing rate: {len(x_flat)/calc_time:.0f} points/second")

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
plt.title('T96 Magnetic Field in X-Z Plane (Y=0)')
plt.axis('equal')
plt.show()