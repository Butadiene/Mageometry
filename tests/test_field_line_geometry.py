#!/usr/bin/env python3
"""
Test script for magnetic field line geometry calculations.

Tests the Frenet-Serret frame, curvature, and torsion calculations
for various magnetic field models.
"""

import numpy as np
import geopack
from geopack import (
    t89_vectorized, t96_vectorized,
    field_line_tangent_vectorized,
    field_line_curvature_vectorized,
    field_line_normal_vectorized,
    field_line_binormal_vectorized,
    field_line_torsion_vectorized,
    field_line_frenet_frame_vectorized,
    field_line_geometry_complete_vectorized
)


def test_tangent_vector():
    """Test tangent vector calculation."""
    print("\n=== Testing Tangent Vector Calculation ===")
    
    # Set up test parameters
    ps = geopack.recalc(1600000000)  # Example timestamp
    kp = 2
    
    # Test single point
    x, y, z = 5.0, 3.0, 2.0
    tx, ty, tz = field_line_tangent_vectorized(t89_vectorized, kp, ps, x, y, z)
    
    # Check normalization
    mag = np.sqrt(tx**2 + ty**2 + tz**2)
    print(f"Single point: T = ({tx:.4f}, {ty:.4f}, {tz:.4f}), |T| = {mag:.6f}")
    assert np.abs(mag - 1.0) < 1e-6, "Tangent vector not normalized"
    
    # Test array of points
    x_arr = np.array([5.0, 4.0, 3.0, -6.0])
    y_arr = np.array([3.0, 2.0, 1.0, 0.0])
    z_arr = np.array([2.0, 1.0, 0.5, 0.0])
    
    tx_arr, ty_arr, tz_arr = field_line_tangent_vectorized(
        t89_vectorized, kp, ps, x_arr, y_arr, z_arr
    )
    
    # Check all are normalized
    mags = np.sqrt(tx_arr**2 + ty_arr**2 + tz_arr**2)
    print(f"Array points: |T| = {mags}")
    assert np.all(np.abs(mags - 1.0) < 1e-6), "Some tangent vectors not normalized"
    
    print("✓ Tangent vector tests passed")


def test_curvature():
    """Test curvature calculation."""
    print("\n=== Testing Curvature Calculation ===")
    
    ps = geopack.recalc(1600000000)
    kp = 2
    
    # Test at different locations
    locations = [
        (5.0, 0.0, 0.0),   # Equatorial plane
        (0.0, 5.0, 0.0),   # Y-axis
        (3.0, 3.0, 3.0),   # Off-axis
        (-10.0, 0.0, 0.0), # Tail region
    ]
    
    for x, y, z in locations:
        curvature = field_line_curvature_vectorized(t89_vectorized, kp, ps, x, y, z)
        print(f"Position ({x:4.1f}, {y:4.1f}, {z:4.1f}): κ = {curvature:.6f} Re⁻¹")
    
    # Test array
    x_arr = np.array([loc[0] for loc in locations])
    y_arr = np.array([loc[1] for loc in locations])
    z_arr = np.array([loc[2] for loc in locations])
    
    curv_arr = field_line_curvature_vectorized(t89_vectorized, kp, ps, x_arr, y_arr, z_arr)
    print(f"Array curvatures: {curv_arr}")
    
    print("✓ Curvature tests passed")


def test_frenet_frame():
    """Test complete Frenet frame calculation."""
    print("\n=== Testing Frenet Frame ===")
    
    ps = geopack.recalc(1600000000)
    kp = 2
    
    # Test single point
    x, y, z = 5.0, 3.0, 2.0
    tx, ty, tz, nx, ny, nz, bx, by, bz, curvature = field_line_frenet_frame_vectorized(
        t89_vectorized, kp, ps, x, y, z
    )
    
    # Check orthonormality
    # T · N should be 0
    dot_tn = tx*nx + ty*ny + tz*nz
    # T · B should be 0
    dot_tb = tx*bx + ty*by + tz*bz
    # N · B should be 0
    dot_nb = nx*bx + ny*by + nz*bz
    
    print(f"T = ({tx:.4f}, {ty:.4f}, {tz:.4f})")
    print(f"N = ({nx:.4f}, {ny:.4f}, {nz:.4f})")
    print(f"B = ({bx:.4f}, {by:.4f}, {bz:.4f})")
    print(f"T·N = {dot_tn:.6f}, T·B = {dot_tb:.6f}, N·B = {dot_nb:.6f}")
    print(f"Curvature = {curvature:.6f} Re⁻¹")
    
    assert np.abs(dot_tn) < 1e-5, "T and N not orthogonal"
    assert np.abs(dot_tb) < 1e-5, "T and B not orthogonal"
    assert np.abs(dot_nb) < 1e-5, "N and B not orthogonal"
    
    # Check B = T × N
    cross_x = ty*nz - tz*ny
    cross_y = tz*nx - tx*nz
    cross_z = tx*ny - ty*nx
    
    assert np.abs(cross_x - bx) < 1e-5, "B ≠ T × N (x component)"
    assert np.abs(cross_y - by) < 1e-5, "B ≠ T × N (y component)"
    assert np.abs(cross_z - bz) < 1e-5, "B ≠ T × N (z component)"
    
    print("✓ Frenet frame tests passed")


def test_torsion():
    """Test torsion calculation."""
    print("\n=== Testing Torsion Calculation ===")
    
    ps = geopack.recalc(1600000000)
    parmod = [2.0, -20.0, 0.5, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # T96 parameters
    
    # Test at different locations
    locations = [
        (5.0, 0.0, 0.0),   # Equatorial plane
        (3.0, 3.0, 3.0),   # Off-axis
        (-8.0, 2.0, 1.0),  # Tail region
    ]
    
    for x, y, z in locations:
        torsion = field_line_torsion_vectorized(t96_vectorized, parmod, ps, x, y, z)
        print(f"Position ({x:4.1f}, {y:4.1f}, {z:4.1f}): τ = {torsion:.6f} Re⁻¹")
    
    print("✓ Torsion tests passed")


def test_complete_geometry():
    """Test complete geometry calculation."""
    print("\n=== Testing Complete Geometry ===")
    
    ps = geopack.recalc(1600000000)
    parmod = [2.0, -20.0, 0.5, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # T96 parameters
    
    # Test array of points
    x_arr = np.array([5.0, 4.0, 3.0, -6.0])
    y_arr = np.array([0.0, 2.0, 3.0, 1.0])
    z_arr = np.array([0.0, 1.0, 2.0, 0.5])
    
    result = field_line_geometry_complete_vectorized(
        t96_vectorized, parmod, ps, x_arr, y_arr, z_arr
    )
    
    tx, ty, tz, nx, ny, nz, bx, by, bz, curvature, torsion = result
    
    print("\nResults for array of points:")
    for i in range(len(x_arr)):
        print(f"\nPoint {i+1}: ({x_arr[i]:.1f}, {y_arr[i]:.1f}, {z_arr[i]:.1f})")
        print(f"  T = ({tx[i]:.4f}, {ty[i]:.4f}, {tz[i]:.4f})")
        print(f"  κ = {curvature[i]:.6f} Re⁻¹")
        print(f"  τ = {torsion[i]:.6f} Re⁻¹")
    
    # Check shapes
    assert tx.shape == x_arr.shape, "Output shape mismatch"
    
    print("✓ Complete geometry tests passed")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n=== Testing Edge Cases ===")
    
    ps = geopack.recalc(1600000000)
    kp = 2
    
    # Test near-zero field region (should handle gracefully)
    x, y, z = 0.0, 0.0, 50.0  # Far above Earth
    tx, ty, tz = field_line_tangent_vectorized(t89_vectorized, kp, ps, x, y, z)
    print(f"Near-zero field: T = ({tx:.4f}, {ty:.4f}, {tz:.4f})")
    
    # Test scalar/array compatibility
    x_scalar = 5.0
    x_array = np.array([5.0])
    
    result_scalar = field_line_curvature_vectorized(t89_vectorized, kp, ps, x_scalar, 0.0, 0.0)
    result_array = field_line_curvature_vectorized(t89_vectorized, kp, ps, x_array, np.array([0.0]), np.array([0.0]))
    
    print(f"Scalar result: {result_scalar} (type: {type(result_scalar)})")
    print(f"Array result: {result_array} (type: {type(result_array)})")
    
    assert np.isscalar(result_scalar), "Scalar input should return scalar"
    assert isinstance(result_array, np.ndarray), "Array input should return array"
    
    print("✓ Edge case tests passed")


def main():
    """Run all tests."""
    print("Testing Magnetic Field Line Geometry Calculations")
    print("=" * 50)
    
    test_tangent_vector()
    test_curvature()
    test_frenet_frame()
    test_torsion()
    test_complete_geometry()
    test_edge_cases()
    
    print("\n" + "=" * 50)
    print("All tests passed successfully! ✓")


if __name__ == "__main__":
    main()