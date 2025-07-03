# Larmor Radius to Radius of Curvature Analysis

This folder contains standalone tools and notebooks for analyzing the ratio of particle Larmor radius to magnetic field line radius of curvature. These tools are independent of the main geopack package and can be used for general particle dynamics studies.

## Contents

### Python Scripts

- **`larmor_curvature_analysis.py`** - Analysis using geopack's magnetic field models
  - Calculates relativistic Larmor radius
  - Uses T96 model for field line curvature
  - Creates comprehensive visualizations
  - Identifies critical regions for particle dynamics

- **`larmor_curvature_standalone.py`** - Completely standalone implementation
  - Includes its own field line curvature calculation
  - Works with any magnetic field model
  - No dependencies on geopack internals
  - Includes simple dipole field example

### Jupyter Notebooks

- **`larmor_curvature_analysis_notebook.ipynb`** - Comprehensive interactive analysis
  - Interactive widgets for parameter exploration
  - Detailed physics explanations
  - Visualization tools
  - Export functions for external use

- **`curvature_larmor_ratio_analysis.ipynb`** - Detailed ratio analysis
  - Energy dependence studies
  - Spatial distribution maps
  - Critical region identification

- **`curvature_scattering_threshold_analysis.ipynb`** - Pitch angle scattering
  - Scattering threshold estimation
  - Non-adiabatic motion regions
  - Particle loss predictions

## Physics Background

The ratio r_L/R_c is fundamental for understanding charged particle motion:

- **r_L << R_c** (ratio < 0.1): Adiabatic motion, first invariant conserved
- **r_L ~ 0.1-0.3 R_c**: Significant pitch angle scattering begins
- **r_L ~ R_c** (ratio ~ 1): Non-adiabatic motion, chaotic trajectories

## Usage

### Standalone Script
```bash
python larmor_curvature_standalone.py
```

### With Geopack Models
```bash
python larmor_curvature_analysis.py
```

### Interactive Notebook
```bash
jupyter notebook larmor_curvature_analysis_notebook.ipynb
```

## Applications

- Radiation belt particle dynamics
- Pitch angle diffusion rates
- Particle precipitation studies
- Magnetic confinement analysis
- Space weather effects on energetic particles

## Custom Field Models

The standalone tools can work with any magnetic field model:

```python
def my_field(x, y, z, **params):
    # Your field calculation
    return Bx, By, Bz

results = analyze_larmor_curvature_ratio(
    x, y, z, my_field,
    energy_keV=100,
    pitch_angle_deg=90
)
```

## Dependencies

- NumPy
- Matplotlib
- IPython/Jupyter (for notebooks)
- ipywidgets (for interactive features)

The standalone script has minimal dependencies and can be easily integrated into other projects.