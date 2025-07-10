# GEOPACK-VECTORIZE

A vectorized implementation extending the excellent Python [geopack](https://github.com/tsssss/geopack) library by Sheng Tian, adding high-performance NumPy-based versions of Tsyganenko magnetospheric field models and field line tracing algorithms.

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

| Component | Scalar Time (1000 points) | Vectorized Time | Speedup |
|-----------|--------------------------|-----------------|---------|
| T89 Model | 1.64 s | 0.012 s | **137x** |
| T96 Model | 15.97 s | 0.135 s | **118x** |
| T01 Model | 37.95 s | 0.250 s | **152x** |
| T04 Model | 34.89 s | 0.278 s | **125x** |
| IGRF | 0.156 s | 0.017 s | **9x** |
| Coordinate Transforms | 0.025 s | 0.001 s | **25x** |
| Field Line Tracing | 2.5 s | 0.08 s | **31x** |

## Installation

### Requirements
- Python 3.7+
- NumPy
- SciPy

### Install from Source
```bash
git clone https://github.com/Butadiene/geopack-vectorize.git
cd geopack-vectorize
pip install -e .
```

### Quick Test
```python
import geopack
from geopack.vectorized import t96_vectorized
import numpy as np

# Set up time
ut = 0  # Unix timestamp
ps = geopack.recalc(ut)

# Vectorized calculation on multiple points
x = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
y = np.zeros(5)
z = np.zeros(5)
parmod = np.array([2.0, -20.0, 0.0, -5.0, 0, 0, 0, 0, 0, 0])

bx, by, bz = t96_vectorized(parmod, ps, x, y, z)
print(f"Magnetic field at x={x[0]}: Bx={bx[0]:.2f}, By={by[0]:.2f}, Bz={bz[0]:.2f} nT")
```

## Usage Examples

### Using the Original Scalar Functions (from geopack)
```python
import geopack
from geopack import t89, t96

# Time setup
ut = 100  # 1970-01-01/00:01:40 UT
ps = geopack.recalc(ut)

# Calculate field at a single point using original scalar function
x, y, z = 5.0, 0.0, 0.0
bx, by, bz = t89(3, ps, x, y, z)  # Kp = 3
print(f"T89 field: Bx={bx:.2f}, By={by:.2f}, Bz={bz:.2f} nT")

# T96 with solar wind parameters
parmod = [2.0, -20.0, 0.0, -5.0, 0, 0, 0, 0, 0, 0]  # [Pdyn, Dst, ByIMF, BzIMF, ...]
bx, by, bz = t96(parmod, ps, x, y, z)
```

### Using the New Vectorized Functions
```python
from geopack.vectorized import t89_vectorized, t96_vectorized
import numpy as np

# OPTION 1: Single point (works just like scalar version)
x, y, z = 5.0, 0.0, 0.0
bx, by, bz = t89_vectorized(3, ps, x, y, z)  # Returns scalars
print(f"T89 vectorized (single): Bx={bx:.2f}, By={by:.2f}, Bz={bz:.2f} nT")

# OPTION 2: Multiple points at once (this is where vectorization shines!)
x_array = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
y_array = np.zeros(5)
z_array = np.zeros(5)
bx_array, by_array, bz_array = t89_vectorized(3, ps, x_array, y_array, z_array)
print(f"Shape of output: {bx_array.shape}")  # (5,)
print(f"First point: Bx={bx_array[0]:.2f} nT")

# OPTION 3: Large arrays for maximum performance
x_large = np.linspace(-10, 10, 10000)
y_large = np.zeros(10000)
z_large = np.zeros(10000)
bx_large, by_large, bz_large = t89_vectorized(3, ps, x_large, y_large, z_large)
# Processes all 10,000 points in ~0.1 seconds!
```

### Vectorized vs Scalar Comparison
```python
import time

# Generate test data
n_points = 1000
x = np.random.uniform(-10, 5, n_points)
y = np.random.uniform(-5, 5, n_points)
z = np.random.uniform(-3, 3, n_points)

# Scalar approach (original geopack)
start = time.time()
bx_scalar = np.zeros(n_points)
by_scalar = np.zeros(n_points)
bz_scalar = np.zeros(n_points)
for i in range(n_points):
    bx_scalar[i], by_scalar[i], bz_scalar[i] = t89(3, ps, x[i], y[i], z[i])
scalar_time = time.time() - start

# Vectorized approach (new)
start = time.time()
bx_vec, by_vec, bz_vec = t89_vectorized(3, ps, x, y, z)
vector_time = time.time() - start

print(f"Scalar time: {scalar_time:.3f} s")
print(f"Vectorized time: {vector_time:.3f} s")
print(f"Speedup: {scalar_time/vector_time:.0f}x")
print(f"Results match: {np.allclose(bx_scalar, bx_vec)}")  # True
```

### Field Line Tracing - Vectorized
```python
from geopack.trace_field_lines_vectorized import trace_vectorized

# Trace a single field line (backward compatible)
x0, y0, z0 = 5.0, 0.0, 0.0
xf, yf, zf, status = trace_vectorized(x0, y0, z0, dir=1, rlim=30)

# Trace multiple field lines in parallel (vectorized)
x0_array = np.array([5.0, 6.0, 7.0, 8.0])
y0_array = np.zeros(4)
z0_array = np.zeros(4)
xf_array, yf_array, zf_array, status_array = trace_vectorized(
    x0_array, y0_array, z0_array, dir=1, rlim=30
)
print(f"Traced {len(x0_array)} field lines simultaneously")
print(f"Status codes: {status_array}")  # 1 = reached boundary, 0 = max steps
```

### Large-Scale Field Mapping with Vectorization
```python
# Create 3D grid
x = np.linspace(-20, 10, 100)
y = np.linspace(-15, 15, 100)
z = np.linspace(-10, 10, 50)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Flatten for vectorized calculation
x_flat = X.ravel()
y_flat = Y.ravel()
z_flat = Z.ravel()

# Calculate field at all 500,000 points using vectorized function
parmod = np.array([2.0, -20.0, 0.0, -5.0, 0, 0, 0, 0, 0, 0])
start = time.time()
bx, by, bz = t96_vectorized(parmod, ps, x_flat, y_flat, z_flat)
print(f"Calculated {len(x_flat):,} points in {time.time()-start:.1f} seconds")

# Reshape results back to 3D
Bx = bx.reshape(X.shape)
By = by.reshape(Y.shape)
Bz = bz.reshape(Z.shape)
```

## Vectorized Components

### Field Models
- `t89_vectorized(iopt, ps, x, y, z)` - T89 Kp-based model
- `t96_vectorized(parmod, ps, x, y, z)` - T96 solar wind parameter model
- `t01_vectorized(parmod, ps, x, y, z)` - T01 storm-time model
- `t04_vectorized(parmod, ps, x, y, z)` - T04 storm-time model

### Field Line Tracing
- `trace_vectorized(x0, y0, z0, dir, rlim, ...)` - Production implementation with boundary interpolation
- Field line geometry calculations (curvature, torsion, Frenet-Serret frame)

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

Comprehensive documentation and example notebooks are available in the repository:

### Tutorial Notebooks (`examples/notebooks/`)
- Coordinate transformation guide
- Field model demonstrations
- Performance benchmarks
- Accuracy validation
- Field line tracing tutorials
- Advanced applications

### Technical Documentation (`docs/`)
- Vectorization principles and guidelines
- Accuracy reports for all models
- Implementation details
- Development guides

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

## Contributing

Contributions are welcome! Areas of interest include:
- GPU acceleration implementations
- Additional field models
- Performance optimizations
- Documentation improvements

Please submit issues or pull requests on the project GitHub page.

## License

This project maintains the MIT License from the original geopack implementation.

## References

- Tsyganenko, N. A. (1995), "Modeling the Earth's magnetospheric magnetic field", J. Geophys. Res.
- Tsyganenko, N. A. (2002), "A model of the near magnetosphere with a dawn-dusk asymmetry", J. Geophys. Res.
- Tsyganenko, N. A. and M. I. Sitnov (2005), "Modeling the dynamics of the inner magnetosphere during strong geomagnetic storms", J. Geophys. Res.
- International Geomagnetic Reference Field: https://www.ngdc.noaa.gov/IAGA/vmod/igrf.html