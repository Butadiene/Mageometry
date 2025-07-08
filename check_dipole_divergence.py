"""
Check divergence of dipole field to understand the issue.
"""
import numpy as np
import matplotlib.pyplot as plt
from geopack import recalc, dip

# Initialize
ut = 0.0
ps = recalc(ut)

# Function to calculate divergence
def calculate_divergence(x, y, z, delta=0.01):
    """Calculate ∇·B using finite differences."""
    # Get field at current point
    bx0, by0, bz0 = dip(x, y, z)
    
    # X derivatives
    bx_plus, _, _ = dip(x + delta, y, z)
    bx_minus, _, _ = dip(x - delta, y, z)
    dbx_dx = (bx_plus - bx_minus) / (2 * delta)
    
    # Y derivatives
    _, by_plus, _ = dip(x, y + delta, z)
    _, by_minus, _ = dip(x, y - delta, z)
    dby_dy = (by_plus - by_minus) / (2 * delta)
    
    # Z derivatives
    _, _, bz_plus = dip(x, y, z + delta)
    _, _, bz_minus = dip(x, y, z - delta)
    dbz_dz = (bz_plus - bz_minus) / (2 * delta)
    
    divergence = dbx_dx + dby_dy + dbz_dz
    return divergence, (bx0, by0, bz0)

# Test at various points
print("Testing divergence at various points:")
print("=" * 60)

test_points = [
    (-3.0, 0.0, 0.0),   # Equatorial
    (-6.0, 0.0, 0.0),   # Farther equatorial  
    (-2.0, 0.0, 4.0),   # The problematic point
    (-2.0, 0.0, 2.0),   # Less extreme
    (-4.0, 0.0, 2.0),   # Middle ground
    (-1.5, 0.0, 0.0),   # Close to Earth
    (-10.0, 0.0, 0.0),  # Far from Earth
]

for x, y, z in test_points:
    r = np.sqrt(x**2 + y**2 + z**2)
    div, (bx, by, bz) = calculate_divergence(x, y, z)
    b_mag = np.sqrt(bx**2 + by**2 + bz**2)
    print(f"Point ({x:4.1f}, {y:4.1f}, {z:4.1f}), r={r:4.2f} Re: "
          f"∇·B = {div:8.3f} nT/Re, |B| = {b_mag:6.1f} nT")

# Create a map of divergence
print("\nCreating divergence map...")
x_grid = np.linspace(-8, -1.5, 50)
z_grid = np.linspace(-4, 4, 40)
X, Z = np.meshgrid(x_grid, z_grid)
Y = np.zeros_like(X)

# Calculate divergence on grid
div_grid = np.zeros_like(X)
b_mag_grid = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        div_grid[i,j], (bx, by, bz) = calculate_divergence(X[i,j], Y[i,j], Z[i,j])
        b_mag_grid[i,j] = np.sqrt(bx**2 + by**2 + bz**2)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Divergence
vmax = 50  # Cap the colorscale
im1 = ax1.contourf(X, Z, np.clip(div_grid, -vmax, vmax), 
                   levels=20, cmap='RdBu_r', extend='both')
ax1.set_title('∇·B (should be ~0 everywhere)')
ax1.set_xlabel('X (Re)')
ax1.set_ylabel('Z (Re)')
ax1.set_aspect('equal')
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('∇·B (nT/Re)')

# Mark test points
for x, y, z in test_points:
    if y == 0:  # Only plot points in the X-Z plane
        ax1.plot(x, z, 'k*', markersize=8)

# Field magnitude for reference
im2 = ax2.contourf(X, Z, b_mag_grid, levels=20, cmap='viridis')
ax2.set_title('|B| (nT)')
ax2.set_xlabel('X (Re)')
ax2.set_ylabel('Z (Re)')
ax2.set_aspect('equal')
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('|B| (nT)')

# Add Earth
earth_theta = np.linspace(0, 2*np.pi, 100)
earth_x = np.cos(earth_theta)
earth_z = np.sin(earth_theta)
ax1.fill(earth_x, earth_z, 'gray', alpha=0.5)
ax2.fill(earth_x, earth_z, 'gray', alpha=0.5)

plt.tight_layout()
plt.savefig('dipole_divergence_check.png', dpi=150)
plt.show()

# Find regions with large divergence
large_div_mask = np.abs(div_grid) > 10
print(f"\nPoints with |∇·B| > 10 nT/Re: {np.sum(large_div_mask)} out of {div_grid.size}")
if np.any(large_div_mask):
    idx = np.unravel_index(np.argmax(np.abs(div_grid)), div_grid.shape)
    max_div_x = X[idx]
    max_div_z = Z[idx]
    max_div = div_grid[idx]
    r_max = np.sqrt(max_div_x**2 + max_div_z**2)
    print(f"Maximum |∇·B| = {np.abs(max_div):.1f} at ({max_div_x:.2f}, 0, {max_div_z:.2f}), r={r_max:.2f} Re")