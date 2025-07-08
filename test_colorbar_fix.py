"""
Test colorbar fix for (∂n/∂b)·b plot
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Create test data similar to the notebook
x = np.linspace(-8, -2, 50)
z = np.linspace(-4, 4, 40)
X, Z = np.meshgrid(x, z)

# Create test data with range similar to (∂n/∂b)·b
data = np.sin(X) * np.cos(Z) * 20  # This will have large values like the real data

print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")

# Create figure with two subplots to compare
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Method 1: Using vmin/vmax in contourf (current approach)
im1 = ax1.contourf(X, Z, data, levels=20, cmap='RdBu_r', vmin=-1, vmax=1, extend='both')
ax1.set_title('Method 1: vmin/vmax in contourf')
ax1.set_aspect('equal')
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
cbar1 = plt.colorbar(im1, cax=cax1)
cbar1.set_label('Value')

# Method 2: Using levels explicitly
levels = np.linspace(-1, 1, 21)
im2 = ax2.contourf(X, Z, data, levels=levels, cmap='RdBu_r', extend='both')
ax2.set_title('Method 2: Explicit levels')
ax2.set_aspect('equal')
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
cbar2 = plt.colorbar(im2, cax=cax2)
cbar2.set_label('Value')

# Add data range text
for ax in [ax1, ax2]:
    ax.text(0.02, 0.98, f"Data: [{data.min():.1f}, {data.max():.1f}]", 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('test_colorbar_methods.png', dpi=150)
plt.show()

# Check which method works better
print("\nColorbar ticks:")
print(f"Method 1 ticks: {cbar1.get_ticks()}")
print(f"Method 2 ticks: {cbar2.get_ticks()}")