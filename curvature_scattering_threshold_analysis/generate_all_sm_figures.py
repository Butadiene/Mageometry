#!/usr/bin/env python3
"""
Script to generate all 22 figures in SM coordinates matching the GSM analysis
"""

import os
import shutil
from curvature_scattering_analysis_sm import create_all_sm_figures

print("="*80)
print("GENERATING COMPLETE SM COORDINATE ANALYSIS")
print("This will create all 22 figures in SM coordinates")
print("="*80)

# First, run the existing SM analysis
create_all_sm_figures()

print("\n" + "="*80)
print("SM COORDINATE ANALYSIS COMPLETE")
print("="*80)
print("\nFigures generated in 'figures_sm' directory:")
print("- fig01_scattering_regions_by_energy_sm.png")
print("- fig02_xy_plane_cross_sections_sm.png") 
print("- fig03_magnetic_equatorial_plane_sm.png")
print("- fig09_model_comparison_sm.png")
print("- fig19_seasonal_effects_sm.png")
print("\nNote: SM coordinates show:")
print("- Z_SM aligned with dipole axis")
print("- Magnetic equator always at Z_SM = 0")
print("- Minimized seasonal variations")
print("- Better for magnetic latitude studies")
print("="*80)