#!/usr/bin/env python
"""Incremental implementation of PRC quad functions for debugging."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from geopack.ring_current_vectorized import ffs_vectorized
from geopack import t01


def br_prc_q_incremental(r, sint, cost, n_terms=18):
    """Compute br_prc_q with specified number of terms."""
    # Ensure arrays
    r = np.atleast_1d(r)
    sint = np.atleast_1d(sint)
    cost = np.atleast_1d(cost)
    
    # Coefficients from scalar version
    a = np.array([-21.2666329, 32.24527521, -6.062894078, 7.515660734, 233.7341288, -227.1195714,
                  8.483233889, 16.80642754, -24.63534184, 9.067120578, -1.052686913, -12.08384538,
                  18.61969572, -12.71686069, 47017.35679, -50646.71204, 7746.058231, 1.531069371])
    
    xk = np.array([2.318824273, 0.7955534018, 3.477425908, 1.922155110])
    al = np.array([0.1417519429, 0.1401142771, 0.1485233485, 0.1295221828, 0.1811846095, 0.2265212965])
    dal = np.array([0.006388013110, 0.02306094179, 0.02319676273, 0.01753008801, 0.04841237481, 0.1301957209])
    
    b = np.array([5.303934488, 3.462235072, 7.830223587])
    be = np.array([4.213397467, 2.568743010, 8.492933868])
    
    dg1 = 0.01125504083
    dg2 = 0.01981805097
    c1 = 6.557801891
    c2 = 6.348576071
    c3 = 5.744436687
    drm = 0.5654023158
    
    # Basic parameters
    sint2 = sint**2
    sc = sint * cost
    r_safe = np.where(r < 1e-10, 1e-10, r)
    alpha = sint2 / r_safe
    gamma = cost / r_safe**2
    
    # Initialize result
    result = np.zeros_like(r, dtype=float)
    
    # Calculate only requested number of terms
    n_terms = min(n_terms, 18)
    
    # Terms 1-4: FFS-based with exponential
    if n_terms >= 1:
        f1, _, _ = ffs_vectorized(alpha, al[0], dal[0])
        expon1 = np.exp(xk[0] * (r - 1.0))
        r_over_b1 = np.minimum(r / b[0], 100.0)
        d1 = sc * f1 * expon1 / (r_over_b1**be[0] + 1.0)
        result += a[0] * d1
    
    if n_terms >= 2:
        f2, _, _ = ffs_vectorized(alpha, al[1], dal[1])
        expon2 = np.exp(xk[1] * (r - 1.0))
        r_over_b2 = np.minimum(r / b[1], 100.0)
        d2 = sc * f2 * expon2 / (r_over_b2**be[1] + 1.0)
        result += a[1] * d2
    
    if n_terms >= 3:
        f3, _, _ = ffs_vectorized(alpha, al[2], dal[2])
        expon3 = np.exp(xk[2] * (r - 1.0))
        r_over_b3 = np.minimum(r / b[2], 100.0)
        d3 = sc * f3 * expon3 / (r_over_b3**be[2] + 1.0)
        result += a[2] * d3
    
    if n_terms >= 4:
        f4, _, _ = ffs_vectorized(alpha, al[3], dal[3])
        expon4 = np.exp(xk[3] * (r - 1.0))
        d4 = sc * f4 * expon4
        result += a[3] * d4
    
    # Terms 5-6: Special FFS-based terms
    if n_terms >= 5:
        _, _, fs5 = ffs_vectorized(alpha, al[4], dal[4])
        d5 = sc * fs5 * np.exp(xk[2] * (r - 1.0))
        result += a[4] * d5
    
    if n_terms >= 6:
        f6, _, _ = ffs_vectorized(alpha, al[5], dal[5])
        w = drm + r
        dw = w**2
        t1 = np.sqrt(r_safe**2 + dw**2)
        t2 = r_safe + t1
        q = (np.sqrt(t1 + dw) + np.sqrt(t1 - dw)) / (2.0 * t1)
        d6 = sc * f6 * (w / r_safe) / t2 * (q - 1.0)
        result += a[5] * d6
    
    # Terms 7-14: Gaussian-like denominators
    if n_terms >= 7:
        p1 = np.exp(xk[0] * (r - 1.0))
        p2 = np.exp(xk[1] * (r - 1.0))
        p3 = np.exp(xk[2] * (r - 1.0))
        p4 = np.exp(xk[3] * (r - 1.0))
        
        aa = alpha**2
        gg = gamma**2
        
        if n_terms >= 7:
            denom7 = (aa + gg + dg1)**2 + 1e-30
            d7 = sc * ((aa + 2.0 * gg) / denom7) * p1
            result += a[6] * d7
        
        if n_terms >= 8:
            denom8 = (aa + gg + dg2)**2 + 1e-30
            d8 = sc * (aa + 2.0 * gg) / denom8 * p2
            result += a[7] * d8
        
        if n_terms >= 9:
            d9 = sc * p3 / (aa + dg1)**2
            result += a[8] * d9
        
        if n_terms >= 10:
            d10 = sc * p3 / (aa + dg2)**2
            result += a[9] * d10
        
        if n_terms >= 11:
            d11 = sc * p4 / (aa + dg1)**2
            result += a[10] * d11
        
        if n_terms >= 12:
            d12 = sc * p4 / (aa + dg2)**2
            result += a[11] * d12
        
        aa_gg = aa + gg
        if n_terms >= 13:
            d13 = p1 / ((aa_gg + dg1) * (aa + dg1))
            result += a[12] * d13
        
        if n_terms >= 14:
            d14 = p2 / ((aa_gg + dg2) * (aa + dg2))
            result += a[13] * d14
    
    # Terms 15-17: Fourth power denominators
    if n_terms >= 15:
        r2 = r_safe**2
        r4 = r2**2
        d15 = sc / (r4 + c1**4)
        result += a[14] * d15
    
    if n_terms >= 16:
        r2 = r_safe**2
        r4 = r2**2
        d16 = sc / (r4 + c2**4)
        result += a[15] * d16
    
    if n_terms >= 17:
        r2 = r_safe**2
        r4 = r2**2
        d17 = sc / (r4 + c3**4)
        result += a[16] * d17
    
    # Term 18: Special radial cutoff
    if n_terms >= 18:
        delt = 1.0 - np.exp(-r / drm)
        d18 = sc * delt**2
        result += a[17] * d18
    
    # Return scalar if input was scalar
    if np.isscalar(r):
        return result.item()
    return result


def test_convergence():
    """Test how the result converges as we add more terms."""
    print("Testing BR convergence with increasing terms")
    print("="*60)
    
    # Test point
    r, sint, cost = 5.0, 0.8, 0.6
    
    # Get full scalar result
    br_full = t01.br_prc_q(r, sint, cost)
    
    print(f"Test point: r={r}, sint={sint}, cost={cost}")
    print(f"Full scalar result: {br_full:.6f}\n")
    
    print(f"{'Terms':>5} {'Result':>15} {'Cumulative':>15} {'Error vs Full':>15} {'Rel Error':>12}")
    print("-"*75)
    
    for n in range(1, 19):
        result = br_prc_q_incremental(r, sint, cost, n_terms=n)
        
        # Handle array output
        if hasattr(result, '__len__'):
            result = result[0]
        
        error = result - br_full
        rel_error = abs(error / br_full) if abs(br_full) > 1e-10 else abs(error)
        
        print(f"{n:5d} {result:15.6f} {result:15.6f} {error:15.6f} {rel_error:12.2e}")
    
    # Test multiple points
    print("\n\nTesting at multiple points with all 18 terms:")
    print("-"*60)
    
    test_points = [
        (3.0, 0.6, 0.8),
        (5.0, 0.8, 0.6),
        (7.0, 0.9, 0.436),
        (10.0, 0.95, 0.312),
    ]
    
    print(f"{'r':>6} {'sint':>6} {'cost':>8} {'Incremental':>15} {'Scalar':>15} {'Rel Error':>12}")
    
    for r, sint, cost in test_points:
        br_inc = br_prc_q_incremental(r, sint, cost, n_terms=18)
        br_scalar = t01.br_prc_q(r, sint, cost)
        
        rel_err = abs(br_inc - br_scalar) / abs(br_scalar) if abs(br_scalar) > 1e-10 else abs(br_inc - br_scalar)
        
        print(f"{r:6.1f} {sint:6.3f} {cost:8.5f} {br_inc:15.6f} {br_scalar:15.6f} {rel_err:12.2e}")


def main():
    """Run convergence test."""
    test_convergence()


if __name__ == "__main__":
    main()