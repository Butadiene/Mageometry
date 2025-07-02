#!/usr/bin/env python
"""
Basic usage examples for the Geopack library.

This script demonstrates how to use both scalar and vectorized versions
of the magnetospheric field models.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import datetime
import geopack

# Set up time and calculate dipole tilt
dt = datetime.datetime(2023, 3, 15, 12, 0, 0)
ut = dt.timestamp()
ps = geopack.recalc(ut)

print(f"Dipole tilt angle: {np.degrees(ps):.2f} degrees")

# Example 1: Scalar T89 model
print("\n=== T89 Model (Scalar) ===")
kp = 3  # Kp index (1-7)
x, y, z = 5.0, 0.0, 0.0  # Position in GSM coordinates (Re)
bx, by, bz = geopack.t89(kp, ps, x, y, z)
print(f"Position: ({x}, {y}, {z}) Re")
print(f"Field: ({bx:.2f}, {by:.2f}, {bz:.2f}) nT")
print(f"Magnitude: {np.sqrt(bx**2 + by**2 + bz**2):.2f} nT")

# Example 2: Vectorized T96 model
print("\n=== T96 Model (Vectorized) ===")
# Model parameters: [Pdyn, Dst, ByIMF, BzIMF, unused...]
parmod = np.array([2.0, -20.0, 0.0, -5.0, 0, 0, 0, 0, 0, 0])

# Create array of positions
n_points = 1000
x = np.linspace(-10, 10, n_points)
y = np.zeros(n_points)
z = np.zeros(n_points)

# Calculate field
bx, by, bz = geopack.t96_vectorized(parmod, ps, x, y, z)

# Show results at a few points
indices = [0, n_points//2, -1]
print("Sample points:")
for i in indices:
    b_mag = np.sqrt(bx[i]**2 + by[i]**2 + bz[i]**2)
    print(f"  X={x[i]:6.1f} Re: B=({bx[i]:7.2f}, {by[i]:7.2f}, {bz[i]:7.2f}) nT, |B|={b_mag:7.2f} nT")

# Example 3: Performance comparison
print("\n=== Performance Comparison ===")
import time

# Test parameters
n_test = 10000
x_test = np.random.uniform(-10, 5, n_test)
y_test = np.random.uniform(-5, 5, n_test)
z_test = np.random.uniform(-3, 3, n_test)

# Time scalar version (sample)
n_sample = 100
t0 = time.time()
for i in range(n_sample):
    _ = geopack.t96(parmod, ps, x_test[i], y_test[i], z_test[i])
t_scalar = (time.time() - t0) * n_test / n_sample

# Time vectorized version
t0 = time.time()
_ = geopack.t96_vectorized(parmod, ps, x_test, y_test, z_test)
t_vector = time.time() - t0

print(f"Scalar (estimated): {t_scalar:.3f} seconds for {n_test} points")
print(f"Vectorized: {t_vector:.3f} seconds for {n_test} points")
print(f"Speedup: {t_scalar/t_vector:.1f}x")
print(f"Throughput: {n_test/t_vector:.0f} points/second")

# Example 4: Coordinate transformations
print("\n=== Coordinate Transformations ===")
# Convert from GSM to GEO coordinates
x_gsm, y_gsm, z_gsm = 5.0, 0.0, 0.0
x_geo, y_geo, z_geo = geopack.geogsm(x_gsm, y_gsm, z_gsm, -1)
print(f"GSM: ({x_gsm}, {y_gsm}, {z_gsm}) Re")
print(f"GEO: ({x_geo:.3f}, {y_geo:.3f}, {z_geo:.3f}) Re")

# Example 5: IGRF model
print("\n=== IGRF Model ===")
# Geographic coordinates
r, theta, phi = 1.0, np.pi/2, 0.0  # Equator, noon
br, btheta, bphi = geopack.igrf_geo(r, theta, phi)
print(f"Position: r={r} Re, theta={np.degrees(theta):.0f}°, phi={np.degrees(phi):.0f}°")
print(f"IGRF field: Br={br:.1f}, Btheta={btheta:.1f}, Bphi={bphi:.1f} nT")