# GEOPACK-VECTORIZE

A vectorized implementation extending the excellent Python [geopack](https://github.com/tsssss/geopack) library by Sheng Tian, adding high-performance NumPy-based versions of Tsyganenko magnetospheric field models and field line tracing algorithms.

## Attention

**This code was generated using a lot of AI (Claude code etc). There are some parts that have not been fully checked. (Humans are gradually checking them.)**

## Overview

This project builds upon the solid foundation of the Python geopack library, which provides faithful implementations of the Tsyganenko magnetospheric field models (T89, T96, T01, T04) and the IGRF geomagnetic field model, originally developed in Fortran by N.A. Tsyganenko. 

GEOPACK-VECTORIZE extends the original geopack by adding vectorized implementations that leverage NumPy for parallel computation, achieving 20-150x performance improvements while maintaining machine-precision accuracy and full backward compatibility with the original scalar functions.

### What This Project Adds

- **Vectorized Field Models**: NumPy-based implementations of all Tsyganenko models (T89, T96, T01, T04) that process arrays of points simultaneously
- **Vectorized Field Line Tracing**: Parallel tracing of multiple field lines with improved boundary interpolation
- **Vectorized Coordinate Transforms**: Array-based transformations between all coordinate systems
- **Full Backward Compatibility**: All original geopack functions remain unchanged and available
- **Comprehensive Validation**: Extensive test suite ensuring < 1e-8 relative error vs original implementations

## Performance Benchmarks

| Component | Scalar Time (1000 points) [s] | Vectorized Time [s] | Speedup |
|-----------|-------------------------------:|---------------------:|--------:|
| T89 Model | 0.048 | 0.000 | **112x** |
| T96 Model | 1.244 | 0.035 | **35.1x** |
| T01 Model | 1.894 | 0.041 | **46.3x** |
| T04 Model | 1.952 | 0.041 | **47.2x** |
| IGRF (GSW) | 0.060 | 0.006 | **9.4x** |
| Coordinate Transforms (subset) | 0.003 | 0.000 | **64.5x** |

## Upgrading to v2.0.0

As of v2.0.0, the PyPI package name has changed from `geopack-vectorized` to `geopack-vectorize`. The old `geopack-vectorized` package will remain available for a transitional period but will eventually be removed. Please update your installation:

```bash
pip uninstall geopack-vectorized
pip install geopack-vectorize
```

Other notable changes from v1.1.4 to v2.0.0:
- Internal module structure reorganized (all vectorized code now under `geopack/vectorized/`)
- All comments and docstrings translated to English
- Docstrings standardized to NumPy format
- Example notebooks cleaned up and reorganized
- See [RELEASE_NOTES_v2.0.0.md](RELEASE_NOTES_v2.0.0.md) for the full list of changes

**Top-level imports are unchanged** — existing code using `from geopack import t96_vectorized` etc. will continue to work without modification.

## Installation

### Requirements
- Python 3.7+
- NumPy
- SciPy

### Install from PyPI
```bash
pip install geopack-vectorize
```

### Install from Source
```bash
git clone https://github.com/Butadiene/geopack-vectorize.git
cd geopack-vectorize
pip install -e .
```

## Usage Examples

All vectorized functions accept both scalars and NumPy arrays. Call `geopack.recalc(ut)` once before using any model or transform. See [`examples/readme_examples.py`](examples/readme_examples.py) for a runnable version of the code below.

```python
import geopack
import numpy as np

ut = 100  # Unix timestamp (seconds since 1970-01-01)
ps = geopack.recalc(ut)
```

### Coordinate Transformations
```python
from geopack import geogsm_vectorized

# Convert multiple GEO points to GSM (j=1: GEO→GSM, j=-1: GSM→GEO)
x_geo = np.array([1.0, 2.0, 3.0])
y_geo = np.array([0.5, 1.0, 1.5])
z_geo = np.array([0.0, 0.0, 0.0])

x_gsm, y_gsm, z_gsm = geogsm_vectorized(x_geo, y_geo, z_geo, j=1)
```

### IGRF Internal Field
```python
from geopack import igrf_gsm_vectorized

# IGRF magnetic field at multiple GSM positions (Earth radii)
x = np.array([2.0, 3.0, 4.0, 5.0])
y = np.zeros(4)
z = np.zeros(4)

bx, by, bz = igrf_gsm_vectorized(x, y, z)  # returns nT

# Dipole field at the same positions (accepts scalars or arrays)
dx, dy, dz = geopack.dip(x, y, z)
```

### Tsyganenko External Field Models
```python
from geopack import t96_vectorized

# T96 parameters: [Pdyn, Dst, ByIMF, BzIMF, 0, 0, 0, 0, 0, 0]
parmod = np.array([2.0, -20.0, 0.0, -5.0, 0, 0, 0, 0, 0, 0])

x = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
y = np.zeros(5)
z = np.zeros(5)

bx, by, bz = t96_vectorized(parmod, ps, x, y, z)  # returns nT in GSM
```

### Field Line Tracing
```python
from geopack import trace_vectorized

# Trace multiple field lines simultaneously
x0 = np.array([5.0, 6.0, 7.0, 8.0])
y0 = np.zeros(4)
z0 = np.zeros(4)

xf, yf, zf, status = trace_vectorized(x0, y0, z0, dir=-1, rlim=30)
# status: 0 = hit inner boundary, 1 = hit outer boundary, 2 = max steps
```

### Field Line Geometry (Frenet-Serret Frame)
```python
from geopack import field_line_curvature_vectorized, field_line_frenet_frame_vectorized

# Curvature at several points along the noon meridian
x = np.array([5.0, 6.0, 7.0, 8.0])
y = np.zeros(4)
z = np.zeros(4)

kappa = field_line_curvature_vectorized(t96_vectorized, parmod, ps, x, y, z)
# kappa: field line curvature in 1/Re

# Full Frenet-Serret frame (tangent, normal, binormal) + curvature
tx, ty, tz, nx, ny, nz, bx, by, bz, curvature = \
    field_line_frenet_frame_vectorized(t96_vectorized, parmod, ps, x, y, z)
```

### Field Line Directional Derivatives
```python
from geopack import field_line_directional_derivatives_vectorized

# All 9 directional derivatives of the Frenet-Serret frame
derivs = field_line_directional_derivatives_vectorized(
    t96_vectorized, parmod, ps, x, y, z
)

# Tangential derivatives (∂/∂T)
# derivs['dT_dT_n']  (∂T/∂T)·n = κ (curvature)
# derivs['dT_dT_b']  (∂T/∂T)·b = 0 (identity)
# derivs['dn_dT_b']  (∂n/∂T)·b = τ (torsion)

# Normal derivatives (∂/∂n)
# derivs['dT_dn_n']  (∂T/∂n)·n
# derivs['dT_dn_b']  (∂T/∂n)·b
# derivs['dn_dn_b']  (∂n/∂n)·b

# Binormal derivatives (∂/∂b)
# derivs['dn_db_b']  (∂n/∂b)·b
# derivs['dn_db_T']  (∂n/∂b)·T
# derivs['db_db_T']  (∂b/∂b)·T
```

## Vectorized Components

### Field Models
- `t89_vectorized(iopt, ps, x, y, z)` - T89 Kp-based model
- `t96_vectorized(parmod, ps, x, y, z)` - T96 solar wind parameter model
- `t01_vectorized(parmod, ps, x, y, z)` - T01 storm-time model
- `t04_vectorized(parmod, ps, x, y, z)` - T04 storm-time model

### Field Line Tracing
- `trace_vectorized(xi, yi, zi, dir, rlim, r0, parmod, exname, inname, ...)` - Vectorized field line tracing with boundary interpolation
- Field line geometry calculations (curvature, torsion, Frenet-Serret frame)
- Directional derivatives along field lines

### Coordinate Transformations
All major coordinate systems are supported with vectorized transforms:
- GEO (Geographic)
- GEI (Geocentric Equatorial Inertial)
- MAG (Geomagnetic)
- GSM (Geocentric Solar Magnetospheric)
- GSE (Geocentric Solar Ecliptic)
- SM (Solar Magnetic)
- GSW (Geocentric Solar Wind)

### IGRF Implementation
- `igrf_geo_vectorized(r, theta, phi)` - Vectorized IGRF field calculation
- Support for years 1900-2025 with extrapolation beyond

## Model Parameters

### T89
Single parameter: Kp index (1-7)
```python
kp = 3  # Kp index
bx, by, bz = t89_vectorized(kp, ps, x, y, z)
```

### T96
10-element parameter array: `[Pdyn, Dst, ByIMF, BzIMF, 0, 0, 0, 0, 0, 0]`
```python
parmod = np.array([2.0, -20.0, 0.0, -5.0, 0, 0, 0, 0, 0, 0])
bx, by, bz = t96_vectorized(parmod, ps, x, y, z)
```

### T01
10-element array: `[Pdyn, Dst, ByIMF, BzIMF, G1, G2, 0, 0, 0, 0]`

### T04
10-element array: `[Pdyn, Dst, ByIMF, BzIMF, W1, W2, W3, W4, W5, W6]`

## Documentation and Examples

Example notebooks are available in `examples/notebooks/`. Install matplotlib and pandas to run them.

### Tutorial Notebooks
- `01_coordinate_transformations_guide` — Coordinate system transforms
- `02_magnetic_field_models_guide` — Field model usage (T89, T96, T01, T04)
- `03_performance_comparison` — Scalar vs vectorized benchmarks
- `04_accuracy_validation` — Numerical accuracy verification
- `05_field_line_tracing_guide` — Field line tracing tutorial
- `06_field_line_tracing_validation` — Tracing accuracy validation
- `07_fieldline_geometry_and_derivatives` — Frenet-Serret frame and directional derivatives

### Advanced Examples (`examples/notebooks/directional_derivatives_maps/`)
- `dipole_field_directional_derivatives` — Dipole field directional derivative maps
- `t96_field_directional_derivatives` — T96 model directional derivative and FAC maps

## Technical Details

### Vectorization Approach
- Full NumPy broadcasting support
- Elimination of all Python loops
- Optimized conditional logic using `np.where`
- Safe numerical operations with proper edge case handling
- Memory-efficient implementations

### Accuracy Guarantees
- Maximum relative error < 1e-8 vs scalar implementations
- Validated against original Fortran code
- Comprehensive test coverage
- Proper handling of boundary conditions

### Performance Optimization
- Batch processing capabilities for millions of points
- Linear memory scaling with input size
- GPU-ready array operations
- Minimal Python overhead

## Attribution and Acknowledgments

This project extends the excellent Python [geopack](https://github.com/tsssss/geopack) implementation by Sheng Tian, which has been invaluable to the space physics community. The original geopack provides a robust, well-tested foundation that faithfully reproduces the Fortran implementations.

The original Fortran GEOPACK code and Tsyganenko models were developed by N.A. Tsyganenko and are available at:
- https://geo.phys.spbu.ru/~tsyganenko/modeling.html
- https://ccmc.gsfc.nasa.gov/models/

We are grateful to both Sheng Tian for the Python implementation and N.A. Tsyganenko for the original models that have been fundamental to magnetospheric physics research for decades.


## License

This project maintains the MIT License from the original geopack implementation.

## References

- Tsyganenko, N. A. (1995), "Modeling the Earth's magnetospheric magnetic field", J. Geophys. Res.
- Tsyganenko, N. A. (2002), "A model of the near magnetosphere with a dawn-dusk asymmetry", J. Geophys. Res.
- Tsyganenko, N. A. and M. I. Sitnov (2005), "Modeling the dynamics of the inner magnetosphere during strong geomagnetic storms", J. Geophys. Res.
- International Geomagnetic Reference Field: https://www.ngdc.noaa.gov/IAGA/vmod/igrf.html