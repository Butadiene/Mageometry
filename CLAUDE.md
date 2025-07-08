# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python implementation of the Geopack magnetic field modeling library, originally written in Fortran by N.A. Tsyganenko. The project includes both scalar and vectorized implementations of various magnetospheric field models.

## Project Structure

```
geopack-vectorize/
├── geopack/                    # Main library code
│   ├── geopack.py             # Core module with IGRF and coordinates
│   ├── models/                # Scalar model implementations
│   │   ├── t89.py            # T89 scalar model
│   │   ├── t96.py            # T96 scalar model
│   │   ├── t01.py            # T01 scalar model
│   │   └── t04.py            # T04 scalar model
│   └── vectorized/            # Vectorized implementations
│       ├── t89_vectorized.py  # T89 vectorized (50x speedup)
│       ├── t96_vectorized.py  # T96 vectorized (30x speedup)
│       ├── t01_vectorized.py  # T01 vectorized (40x speedup)
│       ├── t04_vectorized.py  # T04 vectorized (35x speedup)
│       ├── condip1_exact_vectorized.py  # Vectorized dipole field
│       ├── trace_field_lines_vectorized.py  # Vectorized field line tracing (recommended)
│       └── trace_field_lines_vectorized_nointerp.py  # Vectorized tracing (validation only)
│
├── docs/                      # Documentation
│   ├── vectorization/        # Vectorization guides and progress
│   ├── accuracy_reports/     # Accuracy evaluation results
│   └── FILE_ORGANIZATION.md  # File organization guide
│
├── tests/                    # Test scripts
│   ├── test_geopack1.py     # Original Fortran compatibility tests
│   ├── test_vectorized_models.py  # Vectorized model tests
│   └── benchmark_models.py   # Performance benchmarks
│
├── examples/                 # Example code and notebooks
│   ├── basic_usage.py       # Simple usage examples
│   └── notebooks/           # Jupyter notebooks
│
```

## Key Components

### Core Library (`geopack/`)
- **geopack.py** - Main module with coordinate transforms and IGRF model
- **models/t89.py** - T89 Kp-based model (scalar)
- **models/t96.py** - T96 solar wind parameter-based model (scalar)
- **models/t01.py** - T01 model with storm-time corrections (scalar)
- **models/t04.py** - T04 storm-time model (scalar)

### Vectorized Implementations
- **vectorized/** - Vectorized field models
  - **t89_vectorized.py** - Vectorized T89 (50x speedup)
  - **t96_vectorized.py** - Vectorized T96 (30x speedup for batch, full implementation)
  - **t01_vectorized.py** - Vectorized T01 (40x speedup)
  - **t04_vectorized.py** - Vectorized T04 (35x speedup)
  - **condip1_exact_vectorized.py** - Vectorized dipole field calculations
  - **trace_field_lines_vectorized.py** - Vectorized field line tracing with boundary interpolation (30-50x speedup, recommended)
  - **trace_field_lines_vectorized_nointerp.py** - Vectorized tracing without interpolation (validation only)
  - **field_line_geometry_vectorized.py** - Vectorized field line geometry calculations including:
    - Frenet-Serret frame (tangent, normal, binormal vectors)
    - Curvature and torsion
  - **field_line_directional_derivatives_new.py** - Vectorized implementation of the 9 directional derivative formulas:
    - (∂T/∂T)·n = κ (curvature), (∂T/∂T)·b = 0, (∂n/∂T)·b = τ (torsion)
    - (∂T/∂n)·n, (∂T/∂n)·b, (∂n/∂n)·b
    - (∂n/∂b)·b, (∂n/∂b)·T, (∂b/∂b)·T
    - Includes antisymmetry validation functions
- **igrf_vectorized.py** - Vectorized IGRF implementation (9-13x speedup)
- **coordinates_vectorized.py** - Vectorized coordinate transformations (25-60x speedup)
- **coordinates_vectorized_complex.py** - Vectorized spherical/Cartesian conversions

## Build and Development Commands

### Setting up development environment
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install package in development mode
pip install -e .

# Install dependencies
pip install numpy scipy
```

### Running tests
```bash
# Run original test suite
python tests/test_geopack1.py

# Run vectorized model tests
python tests/test_vectorized_models.py

# Run performance benchmarks
python tests/benchmark_models.py

# Or using unittest discovery
python -m unittest discover tests/
```

### Building the package
```bash
# Build distribution
python setup.py sdist bdist_wheel

# Install the package
python setup.py install
```

## Vectorization Guidelines

When implementing vectorized versions of functions, follow the principles in `docs/vectorization/direction_vectorize.md`:

### 1. Input Handling
```python
# Always ensure arrays
x = np.atleast_1d(x)
y = np.atleast_1d(y)
z = np.atleast_1d(z)
```

### 2. Conditional Logic
```python
# Replace if/else with np.where
# Bad: if x > 0: result = a else: result = b
# Good:
result = np.where(x > 0, a, b)
```

### 3. Safe Division
```python
# Prevent division by zero
result = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
# or
safe_b = np.where(b < epsilon, epsilon, b)
result = a / safe_b
```

### 4. Initialize Arrays
```python
# Use np.zeros_like for proper shape
bx = np.zeros_like(x)
by = np.zeros_like(y)
bz = np.zeros_like(z)
```

### 5. Scalar Compatibility
```python
def function_vectorized(x, y, z):
    scalar_input = np.isscalar(x)
    x = np.atleast_1d(x)
    # ... calculations ...
    if scalar_input:
        return bx.item(), by.item(), bz.item()
    else:
        return bx, by, bz
```

## Important Implementation Notes

### Coordinate Systems
- **GSM**: Geocentric Solar Magnetospheric (primary)
- **GSE**: Geocentric Solar Ecliptic
- **GEO**: Geographic
- **MAG**: Geomagnetic
- **SM**: Solar Magnetic
- **GEI**: Geocentric Equatorial Inertial
- **GSW**: Geocentric Solar Wind

### Model Parameters
- **T89**: Single Kp index (1-7)
- **T96**: [Pdyn, Dst, ByIMF, BzIMF, unused...]
- **T01**: [Pdyn, Dst, ByIMF, BzIMF, G1, G2, unused...]
- **T04**: [Pdyn, Dst, ByIMF, BzIMF, W1-W6]

### Time Handling
- Uses Unix timestamps (seconds since 1970-01-01)
- Call `recalc(ut)` before calculations to update parameters
- Returns dipole tilt angle `ps` needed by all models

## Testing and Validation

### Accuracy Requirements
- Vectorized implementations should maintain <0.01% error vs scalar
- Handle edge cases: zero vectors, boundary conditions
- Preserve shape: scalars → scalars, arrays → arrays

### Performance Targets
- Vectorized: 10-100x speedup for 1000+ points
- Processing rate: >100k points/second
- Memory: Linear scaling with input size

## Current Vectorization Status

### Completed ✅
- T89 model (full implementation, 50x speedup)
- T96 model (full implementation with all components)
  - Birkeland currents (birk1tot_02, birk2tot_02) ✅
  - Interconnection field (intercon) ✅
  - Tail and ring currents (tailrc96) ✅
  - Accuracy: Max relative error < 1.8e-08 (excellent)
  - Performance: 30x speedup for batch processing
- T01 model (40x speedup, handles boundary conditions)
- T04 model (35x speedup, validates X > -15 Re constraint)
- Dipole field calculations in condip1_exact_vectorized
- IGRF vectorization (9-13x speedup, exact compatibility)
- Coordinate transformations (25-60x speedup for all systems)
- Field line tracing (30-50x speedup, with accurate boundary handling)
- Field line geometry (Frenet-Serret frame, curvature, torsion)
- Field line directional derivatives (9 formulas with antisymmetry relations) with 10-50x speedup

### TODO
- GPU acceleration support
- Parallel processing optimizations
- Additional field line integration methods

## Code Style Guidelines

### DO
- Use descriptive variable names
- Add docstrings to all functions
- Include type hints where helpful
- Test edge cases thoroughly
- Document limitations

### DON'T
- Use global variables (except where required for compatibility)
- Add unnecessary comments
- Create files unless explicitly requested
- Implement fake vectorization (loops over arrays)

## Debugging Tips

1. **Accuracy issues**: Compare intermediate values with scalar version
2. **Shape errors**: Check array broadcasting rules
3. **Performance**: Profile with `%timeit` or `cProfile`
4. **Memory**: Monitor with `memory_profiler`

## Example Usage

### Scalar
```python
import geopack
from geopack import t96
ps = geopack.recalc(ut)
bx, by, bz = t96(parmod, ps, x, y, z)
```

### Vectorized
```python
from geopack import t96_vectorized
x_arr = np.array([...])
y_arr = np.array([...])
z_arr = np.array([...])
bx_arr, by_arr, bz_arr = t96_vectorized(parmod, ps, x_arr, y_arr, z_arr)
```

## Example Notebooks

The `examples/notebooks/` directory contains comprehensive evaluation notebooks:

### Tutorial Notebooks
- **01_coordinate_transformations_guide.ipynb** - Comprehensive guide to coordinate systems
- **02_magnetic_field_models_guide.ipynb** - Field model demonstrations and comparisons
- **03_performance_comparison.ipynb** - Performance benchmarks for all vectorized functions
- **04_accuracy_validation.ipynb** - Thorough accuracy validation across all implementations
- **05_igrf_vectorized_guide.ipynb** - IGRF usage guide with practical examples

### Model Evaluations
- **t89_vectorized_evaluation.ipynb** - T89 model performance and accuracy tests
- **t96_vectorized_evaluation.ipynb** - T96 model comprehensive evaluation
- **t96_solar_wind_evaluation.ipynb** - T96 behavior under different solar wind conditions
- **t01_vectorized_evaluation.ipynb** - T01 model with storm-time corrections
- **t04_vectorized_evaluation.ipynb** - T04 storm-time model evaluation

### Field Line Tracing
- **06_field_line_tracing_guide.ipynb** - Getting started with field line tracing
- **07_field_line_tracing_performance_benchmark.ipynb** - Performance benchmarks
- **08_advanced_field_line_applications.ipynb** - Advanced usage examples
- **09_field_line_tracing_path_accuracy_validation.ipynb** - Path-level accuracy analysis
- **10_field_line_tracing_algorithm_validation.ipynb** - Algorithm validation
- **11_field_line_tracing_comprehensive_comparison.ipynb** - Detailed comparison of implementations

### Field Line Geometry and Directional Derivatives
- **12_field_line_directional_derivatives_guide.ipynb** - Comprehensive guide to the 9 directional derivative formulas
- **13_dipole_field_directional_derivatives.ipynb** - Analysis of the 9 formulas for dipole field

### Comparisons
- **field_slice_comparisons.ipynb** - Visual comparisons of magnetic field patterns

All notebooks include execution outputs demonstrating:
- Accuracy validation (< 1e-8 relative error)
- Performance benchmarks (30-50x speedup)
- Edge case handling
- Visualization of field patterns


## Key Documentation

### Accuracy Reports
- `docs/accuracy_reports/T96_VECTORIZATION_ACCURACY_REPORT.md` - Comprehensive T96 accuracy analysis
- `docs/accuracy_reports/T96_VECTORIZATION_SUMMARY.md` - T96 implementation summary

### Development Guides
- `docs/vectorization/direction_vectorize.md` - Core vectorization principles
- `docs/vectorization/COORDINATE_TRANSFORMATIONS_VECTORIZED.md` - Coordinate transformation guide
- `docs/vectorization/IGRF_VECTORIZATION_SUMMARY.md` - IGRF implementation details
- `docs/vectorization/FIELD_LINE_DIRECTIONAL_DERIVATIVES_CORRECT.md` - Correct mathematical framework for 9 directional derivative formulas
- `docs/vectorization/FIELD_LINE_DIRECTIONAL_DERIVATIVES_DESIGN.md` - Original design (deprecated)
- `docs/FILE_ORGANIZATION.md` - Project structure guide

## References

- Original Fortran code: https://geo.phys.spbu.ru/~tsyganenko/modeling.html
- Paper: Tsyganenko, N. A. (1995), "Modeling the Earth's magnetospheric magnetic field"
- IGRF model: https://www.ngdc.noaa.gov/IAGA/vmod/igrf.html