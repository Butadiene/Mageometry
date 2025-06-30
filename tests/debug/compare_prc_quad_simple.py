#!/usr/bin/env python
"""Compare simplified PRC quad with scalar implementation first term."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from geopack.ring_current_vectorized import br_prc_q_vectorized, bt_prc_q_vectorized
from geopack import t01

def extract_first_term_scalar(r, sint, cost):
    """Extract just the first term calculation from scalar br_prc_q."""
    # Parameters from scalar implementation
    a1 = -21.2666329
    xk1 = 2.318824273
    al1 = 0.1417519429
    dal1 = 0.006388013110
    b1 = 5.303934488
    be1 = 4.213397467
    
    # Basic calculations
    sint2 = sint**2
    sc = sint * cost
    alpha = sint2 / r
    
    # FFS function (from scalar)
    sq1 = np.sqrt((alpha + al1)**2 + dal1**2)
    sq2 = np.sqrt((alpha - al1)**2 + dal1**2)
    fa = 2.0 / (sq1 + sq2)
    f = fa * alpha
    
    # First term
    expon1 = np.exp(xk1 * (r - 1.0))
    d1 = sc * f * expon1 / ((r / b1)**be1 + 1.0)
    
    return a1 * d1


def extract_first_term_bt_scalar(r, sint, cost):
    """Extract just the first term calculation from scalar bt_prc_q."""
    # Parameters from scalar implementation
    a1 = 12.74640393
    xk1 = 1.243288286
    al1 = 0.2071721360
    dal1 = 0.05030555417
    b1 = 7.471332374
    be1 = 3.180533613
    
    # Basic calculations
    sint2 = sint**2
    alpha = sint2 / r
    
    # FFS function
    sq1 = np.sqrt((alpha + al1)**2 + dal1**2)
    sq2 = np.sqrt((alpha - al1)**2 + dal1**2)
    fa = 2.0 / (sq1 + sq2)
    f = fa * alpha
    
    # First term (no sc multiplication for bt)
    expon1 = np.exp(xk1 * (r - 1.0))
    d1 = f * expon1 / ((r / b1)**be1 + 1.0)
    
    return a1 * d1


def compare_implementations():
    """Compare vectorized vs scalar first term."""
    print("Comparing first term implementations")
    print("="*60)
    
    # Test points
    test_points = [
        (3.0, 0.6, 0.8),
        (5.0, 0.8, 0.6),
        (7.0, 0.9, 0.436),
        (10.0, 0.95, 0.312),
        (0.1, 0.1, 0.995),
        (5.0, 0.01, 0.99995),
    ]
    
    print("\nBR component comparison:")
    print("-"*60)
    print(f"{'r':>6} {'sint':>6} {'cost':>8} {'Vectorized':>15} {'Scalar 1st':>15} {'Rel Error':>12}")
    
    for r, sint, cost in test_points:
        # Vectorized result
        br_vec = br_prc_q_vectorized(r, sint, cost)
        if hasattr(br_vec, '__len__'):
            br_vec = br_vec[0]
        
        # Scalar first term only
        br_scalar = extract_first_term_scalar(r, sint, cost)
        
        # Relative error
        if abs(br_scalar) > 1e-10:
            rel_err = abs(br_vec - br_scalar) / abs(br_scalar)
        else:
            rel_err = abs(br_vec - br_scalar)
        
        print(f"{r:6.1f} {sint:6.3f} {cost:8.5f} {br_vec:15.6f} {br_scalar:15.6f} {rel_err:12.2e}")
    
    print("\nBT component comparison:")
    print("-"*60)
    print(f"{'r':>6} {'sint':>6} {'cost':>8} {'Vectorized':>15} {'Scalar 1st':>15} {'Rel Error':>12}")
    
    for r, sint, cost in test_points:
        # Vectorized result
        bt_vec = bt_prc_q_vectorized(r, sint, cost)
        if hasattr(bt_vec, '__len__'):
            bt_vec = bt_vec[0]
        
        # Scalar first term only
        bt_scalar = extract_first_term_bt_scalar(r, sint, cost)
        
        # Relative error
        if abs(bt_scalar) > 1e-10:
            rel_err = abs(bt_vec - bt_scalar) / abs(bt_scalar)
        else:
            rel_err = abs(bt_vec - bt_scalar)
        
        print(f"{r:6.1f} {sint:6.3f} {cost:8.5f} {bt_vec:15.6f} {bt_scalar:15.6f} {rel_err:12.2e}")


def test_full_scalar_implementation():
    """Test against full scalar implementation if available."""
    print("\n\nTesting against full scalar implementation")
    print("="*60)
    
    # Check if scalar functions are available
    try:
        # Test a point
        r, sint, cost = 5.0, 0.8, 0.6
        br_full = t01.br_prc_q(r, sint, cost)
        bt_full = t01.bt_prc_q(r, sint, cost)
        
        br_vec = br_prc_q_vectorized(r, sint, cost)
        bt_vec = bt_prc_q_vectorized(r, sint, cost)
        
        if hasattr(br_vec, '__len__'):
            br_vec = br_vec[0]
        if hasattr(bt_vec, '__len__'):
            bt_vec = bt_vec[0]
        
        print(f"\nTest point: r={r}, sint={sint}, cost={cost}")
        print(f"BR - Full scalar: {br_full:.6f}")
        print(f"BR - Vec (1 term): {br_vec:.6f}")
        print(f"BR - Contribution of first term: {br_vec/br_full*100:.1f}%")
        
        print(f"\nBT - Full scalar: {bt_full:.6f}")
        print(f"BT - Vec (1 term): {bt_vec:.6f}")
        print(f"BT - Contribution of first term: {bt_vec/bt_full*100:.1f}%")
        
    except Exception as e:
        print(f"Could not test against full scalar: {e}")


def main():
    """Run comparison tests."""
    compare_implementations()
    test_full_scalar_implementation()


if __name__ == "__main__":
    main()