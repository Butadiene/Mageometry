#!/usr/bin/env python3
"""
Fixed version of fialcos_vectorized with proper per-element recursion tracking.
"""

import numpy as np
from typing import Tuple


def fialcos_vectorized_fixed(r: np.ndarray, theta: np.ndarray, phi: np.ndarray, 
                            n: int, theta0: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Fixed vectorized conical model of Birkeland current field.
    
    This version properly tracks recursion variables per array element.
    
    Parameters
    ----------
    r : ndarray
        Radial coordinate
    theta : ndarray
        Polar angle
    phi : ndarray
        Azimuthal angle
    n : int
        Number of modes to compute (n <= 10)
    theta0 : float
        Angular half-width of the cone
    dt : float
        Angular half-width of the current layer
        
    Returns
    -------
    btheta, bphi : ndarray
        Field components in spherical coordinates (scaled by 800)
    """
    # Ensure arrays
    r = np.atleast_1d(r)
    theta = np.atleast_1d(theta)
    phi = np.atleast_1d(phi)
    
    # Handle scalar input
    scalar_input = r.size == 1
    
    # Initialize output arrays
    shape = r.shape
    btheta = np.zeros(shape)
    bphi = np.zeros(shape)
    
    # Calculate basic quantities
    sinte = np.sin(theta)
    ro = r * sinte
    coste = np.cos(theta)
    sinfi = np.sin(phi)
    cosfi = np.cos(phi)
    
    # tan(theta/2) and cot(theta/2) with safe division
    tg = np.divide(sinte, 1 + coste, out=np.zeros_like(sinte), where=(1 + coste) != 0)
    ctg = np.divide(sinte, 1 - coste, out=np.zeros_like(sinte), where=(1 - coste) != 0)
    
    # Current sheet boundaries
    tetanp = theta0 + dt
    tetanm = theta0 - dt
    
    # Pre-calculate boundary tangents
    tgp = np.tan(tetanp * 0.5)
    tgm = np.tan(tetanm * 0.5)
    tgm2 = tgm * tgm
    tgp2 = tgp * tgp
    
    # Initialize mode arrays
    btn = np.zeros((n, *shape))
    bpn = np.zeros((n, *shape))
    
    # Initialize recursion variables PER ELEMENT
    cosm1 = np.ones(shape)
    sinm1 = np.zeros(shape)
    tm = np.ones(shape)
    
    # These need to be tracked per element!
    tgm2m = np.ones(shape)  # tgm^(2m)
    tgp2m = np.ones(shape)  # tgp^(2m)
    
    # Determine which branch each element is in
    branch1 = theta < tetanm
    branch2 = (theta >= tetanm) & (theta < tetanp)
    branch3 = theta >= tetanp
    
    # Loop over modes
    for m in range(1, n + 1):
        # Update tm
        tm = tm * tg
        
        # Calculate cos(m*phi) and sin(m*phi) using recursion
        ccos = cosm1 * cosfi - sinm1 * sinfi
        ssin = sinm1 * cosfi + cosm1 * sinfi
        cosm1 = ccos
        sinm1 = ssin
        
        # Update recursion variables based on branch
        # Branch 2 and 3 need tgm2m updated
        tgm2m = np.where(branch2 | branch3, tgm2m * tgm2, tgm2m)
        # Only branch 3 needs tgp2m updated
        tgp2m = np.where(branch3, tgp2m * tgp2, tgp2m)
        
        # Initialize t and dtt for this mode
        t = np.zeros(shape)
        dtt = np.zeros(shape)
        
        # Branch 1: theta < tetanm
        if np.any(branch1):
            t[branch1] = tm[branch1]
            dtt[branch1] = 0.5 * m * tm[branch1] * (tg[branch1] + ctg[branch1])
        
        # Branch 2: tetanm <= theta < tetanp
        if np.any(branch2):
            fc = 1 / (tgp - tgm)
            fc1 = 1 / (2 * m + 1)
            tgm2m1 = tgm2m * tgm
            tg21 = 1 + tg * tg
            
            t[branch2] = fc * (tm[branch2] * (tgp - tg[branch2]) + 
                               fc1 * (tm[branch2] * tg[branch2] - tgm2m1[branch2] / tm[branch2]))
            dtt[branch2] = 0.5 * m * fc * tg21[branch2] * (
                tm[branch2] / tg[branch2] * (tgp - tg[branch2]) - 
                fc1 * (tm[branch2] - tgm2m1[branch2] / (tm[branch2] * tg[branch2]))
            )
        
        # Branch 3: theta >= tetanp
        if np.any(branch3):
            fc = 1 / (tgp - tgm)
            fc1 = 1 / (2 * m + 1)
            
            t[branch3] = fc * fc1 * (tgp2m[branch3] * tgp - tgm2m[branch3] * tgm) / tm[branch3]
            dtt[branch3] = -t[branch3] * m * 0.5 * (tg[branch3] + ctg[branch3])
        
        # Calculate field components for this mode
        # Avoid division by zero
        btn[m-1] = np.divide(m * t * ccos, ro, out=np.zeros_like(ro), where=ro != 0)
        bpn[m-1] = np.divide(-dtt * ssin, r, out=np.zeros_like(r), where=r != 0)
    
    # Extract the n-th mode and scale by 800
    btheta = btn[n-1] * 800.0
    bphi = bpn[n-1] * 800.0
    
    # Handle scalar output
    if scalar_input:
        return btheta.item(), bphi.item()
    else:
        return btheta, bphi


# Test the fixed version
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "../..")
    from geopack.t01 import fialcos
    
    print("Testing fixed fialcos_vectorized...")
    
    # Test parameters
    n = 5
    theta0 = 0.7854
    dt = 0.1
    
    # Test points
    r_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    theta_values = np.array([0.5, 0.7, 0.75, 0.8, 0.9])
    phi_values = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    
    # Scalar results
    btheta_scalar = []
    bphi_scalar = []
    for i in range(len(r_values)):
        bt, bp = fialcos(r_values[i], theta_values[i], phi_values[i], n, theta0, dt)
        btheta_scalar.append(bt)
        bphi_scalar.append(bp)
    
    btheta_scalar = np.array(btheta_scalar)
    bphi_scalar = np.array(bphi_scalar)
    
    # Fixed vectorized results
    btheta_vec, bphi_vec = fialcos_vectorized_fixed(r_values, theta_values, phi_values, n, theta0, dt)
    
    # Calculate errors
    abs_err_theta = np.abs(btheta_vec - btheta_scalar)
    abs_err_phi = np.abs(bphi_vec - bphi_scalar)
    
    rel_err_theta = np.divide(abs_err_theta, np.abs(btheta_scalar), 
                             out=abs_err_theta, where=btheta_scalar!=0)
    rel_err_phi = np.divide(abs_err_phi, np.abs(bphi_scalar), 
                           out=abs_err_phi, where=bphi_scalar!=0)
    
    print(f"\nMax relative error theta: {np.max(rel_err_theta):.2e}")
    print(f"Max relative error phi: {np.max(rel_err_phi):.2e}")
    
    if np.max(rel_err_theta) < 1e-10 and np.max(rel_err_phi) < 1e-10:
        print("\nSUCCESS: Fixed version has excellent accuracy!")
    else:
        print("\nWARNING: Still has accuracy issues")
