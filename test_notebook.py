#!/usr/bin/env python3
"""Test and fix the notebook by executing its cells."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import necessary packages
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

# Import geopack
import geopack
from geopack import (
    t89_vectorized, t96_vectorized, t01_vectorized, t04_vectorized,
    field_line_curvature_vectorized
)

# Physical constants
c = 2.99792458e8  # Speed of light (m/s)
me = 9.10938356e-31  # Electron mass (kg)
e = 1.602176634e-19  # Elementary charge (C)
me_c2_keV = 511.0  # Electron rest energy (keV)
Re = 6.371e6  # Earth radius (m)

# Define the critical threshold
CRITICAL_RATIO = 8.0

print("Testing notebook execution...")
print("=" * 60)

# Initialize geopack
ut = 1600000000  # Unix timestamp
ps = geopack.recalc(ut)

# Model parameters
parmod_quiet = [1.0, -5.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
parmod_moderate = [3.0, -30.0, 1.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
parmod_storm = [10.0, -100.0, 5.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

print(f"Initialized with dipole tilt: {np.degrees(ps):.2f}°")

# Test helper functions
def calculate_larmor_radius(energy_keV, B_nT, pitch_angle_deg=90):
    """Calculate the Larmor radius for an electron."""
    E_k = energy_keV * 1000 * e  # Convert keV to Joules
    B = B_nT * 1e-9  # Convert nT to Tesla
    
    gamma = 1 + E_k / (me * c**2)
    beta = np.sqrt(1 - 1/gamma**2)
    v = beta * c
    
    alpha = np.radians(pitch_angle_deg)
    v_perp = v * np.sin(alpha)
    
    RL = gamma * me * v_perp / (e * B)
    return RL

def calculate_curvature_radius(model_func, parmod, ps, x, y, z):
    """Calculate radius of curvature and magnetic field strength."""
    kappa = field_line_curvature_vectorized(model_func, parmod, ps, x, y, z)
    Rc = np.where(kappa > 1e-10, 1.0 / kappa, 1e10)
    
    bx, by, bz = model_func(parmod, ps, x, y, z)
    B = np.sqrt(bx**2 + by**2 + bz**2)
    
    return Rc, B

# Test Analysis 1
print("\nTesting Analysis 1: Energy-dependent plots...")
try:
    energies = [10, 30, 100, 300, 1000]  # keV
    
    # Create grid for meridian plane
    x_grid = np.linspace(-15, 10, 51)  # Reduced resolution for testing
    z_grid = np.linspace(-10, 10, 41)
    X_mer, Z_mer = np.meshgrid(x_grid, z_grid)
    Y_mer = np.zeros_like(X_mer)
    
    x_flat_mer = X_mer.flatten()
    y_flat_mer = Y_mer.flatten()
    z_flat_mer = Z_mer.flatten()
    
    # Test for one energy
    energy = 100  # keV
    Rc_Re_mer, B_nT_mer = calculate_curvature_radius(t96_vectorized, parmod_moderate, ps, 
                                                     x_flat_mer, y_flat_mer, z_flat_mer)
    Rc_m_mer = Rc_Re_mer * Re
    
    # Calculate Larmor radius
    RL_m_mer = calculate_larmor_radius(energy, B_nT_mer, pitch_angle_deg=90)
    ratio_mer = Rc_m_mer / RL_m_mer
    
    # Clean up extreme values
    ratio_mer = np.where(ratio_mer > 1000, 1000, ratio_mer)
    ratio_mer = np.where(ratio_mer < 0.1, 0.1, ratio_mer)
    
    # Calculate scattering fraction
    scatter_frac = np.sum(ratio_mer < CRITICAL_RATIO) / len(ratio_mer) * 100
    print(f"  {energy} keV: {scatter_frac:.1f}% scattering in meridian plane")
    
except Exception as e:
    print(f"  ERROR in Analysis 1: {e}")
    import traceback
    traceback.print_exc()

# Test Analysis 2
print("\nTesting Analysis 2: Magnetic equatorial plane...")
try:
    # Create grid for equatorial plane
    x_eq = np.linspace(-12, 10, 50)
    y_eq = np.linspace(-11, 11, 50)
    X_eq, Y_eq = np.meshgrid(x_eq, y_eq)
    Z_eq = np.zeros_like(X_eq)
    
    # Calculate ratio
    Rc_Re_eq, B_nT_eq = calculate_curvature_radius(t96_vectorized, parmod_moderate, ps,
                                                   X_eq.flatten(), Y_eq.flatten(), Z_eq.flatten())
    RL_m_eq = calculate_larmor_radius(100, B_nT_eq, pitch_angle_deg=90)
    ratio_eq = Rc_Re_eq * Re / RL_m_eq
    ratio_eq = np.where(ratio_eq > 1000, 1000, ratio_eq)
    ratio_eq = np.where(ratio_eq < 0.1, 0.1, ratio_eq)
    
    scatter_frac_eq = np.sum(ratio_eq < CRITICAL_RATIO) / len(ratio_eq) * 100
    print(f"  100 keV: {scatter_frac_eq:.1f}% scattering in equatorial plane")
    
except Exception as e:
    print(f"  ERROR in Analysis 2: {e}")
    import traceback
    traceback.print_exc()

# Test critical energy calculation
print("\nTesting critical energy calculation...")
try:
    # Test at a single point
    x, y, z = 5.0, 0.0, 0.0
    Rc_Re, B_nT = calculate_curvature_radius(t96_vectorized, parmod_moderate, ps, x, y, z)
    Rc_m = Rc_Re * Re
    
    # Energy range for searching
    E_search = np.logspace(0, 3.5, 50)  # 1 keV to ~3 MeV
    
    # Find critical energy
    critical_energy = None
    for E in E_search:
        RL_m = calculate_larmor_radius(E, B_nT, pitch_angle_deg=90)
        ratio = Rc_m / RL_m
        if ratio < CRITICAL_RATIO:
            critical_energy = E
            break
    
    if critical_energy:
        print(f"  Critical energy at ({x}, {y}, {z}) Re: {critical_energy:.1f} keV")
    else:
        print(f"  No critical energy found (always adiabatic)")
        
except Exception as e:
    print(f"  ERROR in critical energy calculation: {e}")
    import traceback
    traceback.print_exc()

print("\nAll tests completed!")
print("=" * 60)