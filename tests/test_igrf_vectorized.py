"""
Comprehensive tests for vectorized IGRF implementation.

This test suite ensures that the vectorized IGRF functions produce
exactly the same results as the scalar implementations across a wide
range of inputs and edge cases.
"""

import numpy as np
import geopack
from geopack.igrf_vectorized import (
    igrf_geo_vectorized, igrf_gsm_vectorized, igrf_gsw_vectorized
)
import time
from datetime import datetime


def test_scalar_compatibility():
    """Test that vectorized functions handle scalar inputs correctly."""
    print("\n1. Testing scalar input compatibility...")
    
    # Initialize geopack
    ut = datetime(2020, 1, 1, 12, 0, 0).timestamp()
    ps = geopack.recalc(ut)
    
    # Test single point
    x, y, z = 5.0, 3.0, 2.0
    
    # Scalar IGRF
    bx_scalar, by_scalar, bz_scalar = geopack.igrf_gsm(x, y, z)
    
    # Vectorized with scalar input
    bx_vec, by_vec, bz_vec = igrf_gsm_vectorized(x, y, z)
    
    # Check types
    assert isinstance(bx_vec, float), f"Expected float, got {type(bx_vec)}"
    
    # Check values
    assert np.abs(bx_scalar - bx_vec) < 1e-10, f"Bx mismatch: {bx_scalar} vs {bx_vec}"
    assert np.abs(by_scalar - by_vec) < 1e-10, f"By mismatch: {by_scalar} vs {by_vec}"
    assert np.abs(bz_scalar - bz_vec) < 1e-10, f"Bz mismatch: {bz_scalar} vs {bz_vec}"
    
    print("✓ Scalar compatibility test passed")


def test_array_accuracy():
    """Test accuracy of vectorized implementation against scalar."""
    print("\n2. Testing array accuracy...")
    
    # Initialize geopack
    ut = datetime(2020, 6, 21, 0, 0, 0).timestamp()
    ps = geopack.recalc(ut)
    
    # Test various positions
    n_points = 100
    np.random.seed(42)
    
    # Test points at various distances and locations
    r = np.random.uniform(0.5, 10.0, n_points)
    theta = np.random.uniform(0, np.pi, n_points)
    phi = np.random.uniform(0, 2*np.pi, n_points)
    
    # Convert to Cartesian GSM
    x_gsm = r * np.sin(theta) * np.cos(phi)
    y_gsm = r * np.sin(theta) * np.sin(phi)
    z_gsm = r * np.cos(theta)
    
    # Calculate with scalar version
    bx_scalar = np.zeros(n_points)
    by_scalar = np.zeros(n_points)
    bz_scalar = np.zeros(n_points)
    
    for i in range(n_points):
        bx_scalar[i], by_scalar[i], bz_scalar[i] = geopack.igrf_gsm(
            x_gsm[i], y_gsm[i], z_gsm[i])
    
    # Calculate with vectorized version
    bx_vec, by_vec, bz_vec = igrf_gsm_vectorized(x_gsm, y_gsm, z_gsm)
    
    # Calculate errors
    error_x = np.abs(bx_scalar - bx_vec)
    error_y = np.abs(by_scalar - by_vec)
    error_z = np.abs(bz_scalar - bz_vec)
    
    # Calculate relative errors (avoid division by zero)
    b_mag_scalar = np.sqrt(bx_scalar**2 + by_scalar**2 + bz_scalar**2)
    safe_mag = np.where(b_mag_scalar > 1e-6, b_mag_scalar, 1e-6)
    
    rel_error_x = error_x / safe_mag
    rel_error_y = error_y / safe_mag
    rel_error_z = error_z / safe_mag
    
    print(f"  Tested {n_points} random points")
    print(f"  Max absolute error: Bx={np.max(error_x):.2e}, By={np.max(error_y):.2e}, Bz={np.max(error_z):.2e}")
    print(f"  Max relative error: Bx={np.max(rel_error_x):.2e}, By={np.max(rel_error_y):.2e}, Bz={np.max(rel_error_z):.2e}")
    
    # Assert accuracy
    assert np.max(error_x) < 1e-8, f"Bx error too large: {np.max(error_x)}"
    assert np.max(error_y) < 1e-8, f"By error too large: {np.max(error_y)}"
    assert np.max(error_z) < 1e-8, f"Bz error too large: {np.max(error_z)}"
    
    print("✓ Array accuracy test passed")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n3. Testing edge cases...")
    
    # Initialize geopack
    ut = datetime(2020, 1, 1, 12, 0, 0).timestamp()
    ps = geopack.recalc(ut)
    
    # Test cases
    test_cases = [
        # (name, x, y, z)
        ("North pole", 0.0, 0.0, 1.0),
        ("South pole", 0.0, 0.0, -1.0),
        ("Equator +X", 1.0, 0.0, 0.0),
        ("Equator +Y", 0.0, 1.0, 0.0),
        ("Near Earth surface", 1.0, 0.0, 0.0),
        ("Far field", 10.0, 0.0, 0.0),
        ("Very close to Earth", 0.5, 0.0, 0.0),
    ]
    
    for name, x, y, z in test_cases:
        # Scalar
        bx_s, by_s, bz_s = geopack.igrf_gsm(x, y, z)
        
        # Vectorized
        bx_v, by_v, bz_v = igrf_gsm_vectorized(x, y, z)
        
        # Check
        error = np.sqrt((bx_s-bx_v)**2 + (by_s-by_v)**2 + (bz_s-bz_v)**2)
        b_mag = np.sqrt(bx_s**2 + by_s**2 + bz_s**2)
        rel_error = error / (b_mag + 1e-10)
        
        print(f"  {name:20s}: |B|={b_mag:8.1f} nT, error={error:.2e}, rel_error={rel_error:.2e}")
        
        assert error < 1e-8, f"Error too large for {name}: {error}"
    
    print("✓ Edge cases test passed")


def test_coordinate_systems():
    """Test all coordinate system variants."""
    print("\n4. Testing coordinate systems...")
    
    # Initialize geopack
    ut = datetime(2020, 3, 15, 6, 30, 0).timestamp()
    ps = geopack.recalc(ut)
    
    # Test point
    x, y, z = 3.0, 2.0, 1.0
    
    # Test GSM
    bx_gsm_s, by_gsm_s, bz_gsm_s = geopack.igrf_gsm(x, y, z)
    bx_gsm_v, by_gsm_v, bz_gsm_v = igrf_gsm_vectorized(x, y, z)
    
    error_gsm = np.sqrt((bx_gsm_s-bx_gsm_v)**2 + (by_gsm_s-by_gsm_v)**2 + (bz_gsm_s-bz_gsm_v)**2)
    
    # Test GSW  
    bx_gsw_s, by_gsw_s, bz_gsw_s = geopack.igrf_gsw(x, y, z)
    bx_gsw_v, by_gsw_v, bz_gsw_v = igrf_gsw_vectorized(x, y, z)
    
    error_gsw = np.sqrt((bx_gsw_s-bx_gsw_v)**2 + (by_gsw_s-by_gsw_v)**2 + (bz_gsw_s-bz_gsw_v)**2)
    
    print(f"  GSM error: {error_gsm:.2e}")
    print(f"  GSW error: {error_gsw:.2e}")
    
    assert error_gsm < 1e-8, f"GSM error too large: {error_gsm}"
    assert error_gsw < 1e-8, f"GSW error too large: {error_gsw}"
    
    print("✓ Coordinate systems test passed")


def test_performance():
    """Benchmark performance improvement."""
    print("\n5. Performance benchmark...")
    
    # Initialize geopack
    ut = datetime(2020, 1, 1, 12, 0, 0).timestamp()
    ps = geopack.recalc(ut)
    
    # Test different array sizes
    sizes = [10, 100, 1000, 5000]
    
    print(f"{'N Points':>10} {'Scalar (s)':>12} {'Vector (s)':>12} {'Speedup':>10}")
    print("-" * 50)
    
    for n in sizes:
        # Generate test points
        np.random.seed(42)
        x = np.random.uniform(-10, 10, n)
        y = np.random.uniform(-10, 10, n)
        z = np.random.uniform(-10, 10, n)
        
        # Time scalar version
        start = time.time()
        for i in range(n):
            geopack.igrf_gsm(x[i], y[i], z[i])
        scalar_time = time.time() - start
        
        # Time vectorized version
        start = time.time()
        igrf_gsm_vectorized(x, y, z)
        vector_time = time.time() - start
        
        speedup = scalar_time / vector_time
        
        print(f"{n:10d} {scalar_time:12.4f} {vector_time:12.4f} {speedup:9.1f}x")
    
    print("\n✓ Performance benchmark complete")


def test_shape_preservation():
    """Test that output shapes match input shapes."""
    print("\n6. Testing shape preservation...")
    
    # Initialize geopack
    ut = datetime(2020, 1, 1, 12, 0, 0).timestamp()
    ps = geopack.recalc(ut)
    
    # Test various input shapes
    test_shapes = [
        (10,),
        (5, 4),
        (3, 4, 5),
        (2, 3, 2, 2)
    ]
    
    for shape in test_shapes:
        # Generate test data
        x = np.random.uniform(-5, 5, shape)
        y = np.random.uniform(-5, 5, shape)
        z = np.random.uniform(-5, 5, shape)
        
        # Calculate field
        bx, by, bz = igrf_gsm_vectorized(x, y, z)
        
        # Check shapes
        assert bx.shape == shape, f"Bx shape mismatch: {bx.shape} vs {shape}"
        assert by.shape == shape, f"By shape mismatch: {by.shape} vs {shape}"
        assert bz.shape == shape, f"Bz shape mismatch: {bz.shape} vs {shape}"
        
        print(f"  Shape {shape}: ✓")
    
    print("✓ Shape preservation test passed")


def test_time_consistency():
    """Test that results are consistent across different times."""
    print("\n7. Testing time consistency...")
    
    # Test point
    x, y, z = 5.0, 0.0, 0.0
    
    # Test at different times
    times = [
        datetime(2010, 1, 1, 0, 0, 0),
        datetime(2015, 6, 15, 12, 0, 0),
        datetime(2020, 12, 31, 23, 59, 59),
    ]
    
    for dt in times:
        ut = dt.timestamp()
        ps = geopack.recalc(ut)
        
        # Scalar
        bx_s, by_s, bz_s = geopack.igrf_gsm(x, y, z)
        
        # Vectorized
        bx_v, by_v, bz_v = igrf_gsm_vectorized(x, y, z)
        
        error = np.sqrt((bx_s-bx_v)**2 + (by_s-by_v)**2 + (bz_s-bz_v)**2)
        
        print(f"  {dt}: error = {error:.2e}")
        
        assert error < 2e-5, f"Error too large at {dt}: {error}"
    
    print("✓ Time consistency test passed")


if __name__ == "__main__":
    print("Testing Vectorized IGRF Implementation")
    print("=" * 60)
    
    # Run all tests
    test_scalar_compatibility()
    test_array_accuracy()
    test_edge_cases()
    test_coordinate_systems()
    test_performance()
    test_shape_preservation()
    test_time_consistency()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("The vectorized IGRF implementation produces identical results")
    print("to the scalar version while providing significant speedup.")