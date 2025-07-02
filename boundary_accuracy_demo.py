"""
Demonstrate why accurate boundary interpolation matters
"""
import numpy as np
import matplotlib.pyplot as plt
import geopack
from geopack.trace_vectorized import trace_vectorized
from geopack.trace_vectorized_no_interp import trace_vectorized_no_interp

# Initialize
ut = 100.0
ps = geopack.recalc(ut)

# Create a grid of starting points in the tail
x_start = np.linspace(-5, -25, 20)
y_start = np.zeros_like(x_start)
z_start = np.full_like(x_start, 2.0)

# Trace to boundary with both methods
endpoints_interp = []
endpoints_no_interp = []

for x0, y0, z0 in zip(x_start, y_start, z_start):
    # With interpolation
    xf, yf, zf, status = trace_vectorized(x0, y0, z0, dir=1, rlim=30)
    if status == 1:  # Hit outer boundary
        endpoints_interp.append((xf, yf, zf))
    
    # Without interpolation
    xf, yf, zf, status = trace_vectorized_no_interp(x0, y0, z0, dir=1, rlim=30)
    if status == 1:  # Hit outer boundary
        endpoints_no_interp.append((xf, yf, zf))

# Convert to arrays
endpoints_interp = np.array(endpoints_interp)
endpoints_no_interp = np.array(endpoints_no_interp)

# Calculate radii
r_interp = np.sqrt(endpoints_interp[:,0]**2 + endpoints_interp[:,1]**2 + endpoints_interp[:,2]**2)
r_no_interp = np.sqrt(endpoints_no_interp[:,0]**2 + endpoints_no_interp[:,1]**2 + endpoints_no_interp[:,2]**2)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Boundary crossing positions
ax1.set_title('Magnetopause Crossing Positions (X-Z plane)')
ax1.scatter(endpoints_interp[:,0], endpoints_interp[:,2], 
           label=f'With interpolation (r={np.mean(r_interp):.3f}±{np.std(r_interp):.3f})', 
           alpha=0.7, s=50)
ax1.scatter(endpoints_no_interp[:,0], endpoints_no_interp[:,2], 
           label=f'Without interpolation (r={np.mean(r_no_interp):.3f}±{np.std(r_no_interp):.3f})', 
           alpha=0.7, s=50)

# Add boundary circle
theta = np.linspace(0, 2*np.pi, 100)
x_circle = 30 * np.cos(theta)
z_circle = 30 * np.sin(theta)
ax1.plot(x_circle, z_circle, 'k--', label='r=30 boundary', alpha=0.5)

ax1.set_xlabel('X (Re)')
ax1.set_ylabel('Z (Re)')
ax1.set_xlim(-35, 35)
ax1.set_ylim(-35, 35)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Plot 2: Radial distance distribution
ax2.set_title('Distribution of Boundary Crossing Radii')
ax2.hist(r_interp, bins=20, alpha=0.7, label='With interpolation', edgecolor='black')
ax2.hist(r_no_interp, bins=20, alpha=0.7, label='Without interpolation', edgecolor='black')
ax2.axvline(30.0, color='red', linestyle='--', label='Exact boundary (30 Re)')
ax2.set_xlabel('Radial Distance (Re)')
ax2.set_ylabel('Count')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('boundary_accuracy_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Quantitative comparison
print("\nBOUNDARY CROSSING ACCURACY STATISTICS")
print("=" * 50)
print(f"Number of traces that hit boundary: {len(endpoints_interp)}")
print(f"\nWith interpolation:")
print(f"  Mean radius: {np.mean(r_interp):.6f} Re")
print(f"  Std deviation: {np.std(r_interp):.6f} Re")
print(f"  Min radius: {np.min(r_interp):.6f} Re")
print(f"  Max radius: {np.max(r_interp):.6f} Re")
print(f"  Mean error from 30.0: {np.mean(np.abs(r_interp - 30.0)):.6f} Re")

print(f"\nWithout interpolation:")
print(f"  Mean radius: {np.mean(r_no_interp):.6f} Re")
print(f"  Std deviation: {np.std(r_no_interp):.6f} Re")
print(f"  Min radius: {np.min(r_no_interp):.6f} Re")
print(f"  Max radius: {np.max(r_no_interp):.6f} Re")
print(f"  Mean error from 30.0: {np.mean(np.abs(r_no_interp - 30.0)):.6f} Re")

print(f"\nAccuracy improvement factor: {np.mean(np.abs(r_no_interp - 30.0)) / np.mean(np.abs(r_interp - 30.0)):.0f}x")

# Physical implications
print("\n\nPHYSICAL IMPLICATIONS")
print("=" * 50)
print("For magnetospheric physics studies:")
print(f"- Magnetopause standoff distance error with interpolation: {np.mean(np.abs(r_interp - 30.0)) * 6371.2:.1f} km")
print(f"- Magnetopause standoff distance error without: {np.mean(np.abs(r_no_interp - 30.0)) * 6371.2:.1f} km")
print(f"\nFor a typical magnetopause at 10 Re:")
print(f"- Position uncertainty with interpolation: {np.mean(np.abs(r_interp - 30.0)) / 30.0 * 10 * 6371.2:.1f} km")
print(f"- Position uncertainty without: {np.mean(np.abs(r_no_interp - 30.0)) / 30.0 * 10 * 6371.2:.1f} km")
print(f"\nThis affects:")
print("- Solar wind entry calculations")
print("- Particle precipitation mapping")
print("- Magnetopause reconnection studies")
print("- Space weather forecasting accuracy")