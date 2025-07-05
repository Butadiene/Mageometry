# Usage Guide for Curvature Scattering Analysis

## Quick Start

To run the complete analysis and generate all figures:

```bash
cd curvature_scattering_threshold_analysis
python run_analysis.py
```

This will:
1. Calculate Rc/RL ratios across the magnetosphere
2. Generate 22 comprehensive figures
3. Save all figures to the `figures/` directory
4. Print progress and statistics to the console

## Expected Runtime

The full analysis typically takes 2-3 minutes depending on your system, as it:
- Processes multiple energy levels (10 keV - 1 MeV)
- Analyzes different geomagnetic conditions
- Compares 4 different magnetospheric models
- Generates high-resolution plots

## Output Files

After running, you'll find:

### Figures Directory
- `fig01_scattering_regions_by_energy.png` - Energy dependence analysis
- `fig02_xy_plane_cross_sections.png` - 3D structure at different Z heights
- `fig03_magnetic_equatorial_plane.png` - MLT dependence and drift paths
- `fig04_critical_energy_maps.png` - Critical energy thresholds
- `fig05_pitch_angle_effects.png` - Pitch angle sensitivity
- `fig06_storm_evolution_spatial.png` - Storm phase spatial patterns
- `fig07_storm_evolution_temporal.png` - Temporal evolution
- `fig08_t96_parameter_sensitivity.png` - T96 model parameters
- `fig09_model_comparison.png` - Comprehensive model comparison
- `fig10_summary.png` - Summary figure with key findings
- `fig11_3d_field_lines.png` - 3D field lines from scattering regions
- `fig12_3d_volume_rendering.png` - 3D volume visualization of scattering regions
- `fig13_t96_xy_planes.png` - T96 XY plane at different Z heights
- `fig14_t01_xy_planes.png` - T01 XY plane at different Z heights
- `fig15_t04_xy_planes.png` - T04 XY plane at different Z heights
- `fig16_t01_storm_evolution.png` - T01 storm evolution analysis
- `fig17_t04_storm_progression.png` - T04 storm progression analysis
- `fig18_model_comparison_xy_planes.png` - Model comparison XY planes
- `fig19_t96_seasonal_tilt.png` - T96 seasonal dipole tilt effects (Z = 0.0 to 1.4 Re)
- `fig20_seasonal_evolution.png` - Seasonal evolution throughout the year
- `fig21_seasonal_mlt_distribution.png` - MLT distribution for different seasons
- `fig22_seasonal_xz_planes.png` - XZ plane (Y = 0) for different seasonal tilts

### Main Report
- `README.md` - Complete analysis report with all results

## Running Individual Analyses

To run specific analyses only, you can import and call individual functions:

```python
from curvature_scattering_analysis import (
    analyze_scattering_regions_by_energy,
    analyze_magnetic_equatorial_plane,
    analyze_storm_evolution
)

# Run specific analysis
analyze_scattering_regions_by_energy()
```

## Customizing Parameters

To modify analysis parameters, edit the constants in `curvature_scattering_analysis.py`:

```python
# Critical threshold
CRITICAL_RATIO = 8.0  # Default Rc/RL threshold

# Model parameters
parmod_quiet = [1.0, -5.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
parmod_moderate = [3.0, -30.0, 1.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
parmod_storm = [10.0, -100.0, 5.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Energy levels to analyze
energies = [10, 30, 100, 300, 1000]  # keV
```

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- geopack-vectorize (must be installed)

## Troubleshooting

If you encounter import errors:
1. Ensure geopack is properly installed
2. Run from the `curvature_scattering_threshold_analysis` directory
3. Check that all vectorized modules are available

For memory issues with large grids:
- Reduce grid resolution in the analysis functions
- Process fewer energy levels at once
- Close other applications to free memory