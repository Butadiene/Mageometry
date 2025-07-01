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
│       └── condip1_exact_vectorized.py  # Vectorized dipole field
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
└── examples/                 # Example code and notebooks
    ├── basic_usage.py       # Simple usage examples
    ├── field_line_tracing.py # Field line tracing demo
    └── notebooks/           # Jupyter notebooks
```

## Key Components

### Core Library (`geopack/`)
- **geopack.py** - Main module with coordinate transforms and IGRF model
- **models/t89.py** - T89 Kp-based model (scalar)
- **models/t96.py** - T96 solar wind parameter-based model (scalar)
- **models/t01.py** - T01 model with storm-time corrections (scalar)
- **models/t04.py** - T04 storm-time model (scalar)

### Vectorized Implementations (`geopack/vectorized/`)
- **t89_vectorized.py** - Vectorized T89 (50x speedup)
- **t96_vectorized.py** - Vectorized T96 (30x speedup for batch, full implementation)
- **t01_vectorized.py** - Vectorized T01 (40x speedup)
- **t04_vectorized.py** - Vectorized T04 (35x speedup)
- **condip1_exact_vectorized.py** - Vectorized dipole field calculations

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

### TODO
- IGRF vectorization
- Additional coordinate transform optimizations
- Field line tracing optimizations

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

## Key Documentation

### Accuracy Reports
- `docs/accuracy_reports/T96_VECTORIZATION_ACCURACY_REPORT.md` - Comprehensive T96 accuracy analysis
- `docs/accuracy_reports/T96_VECTORIZATION_SUMMARY.md` - T96 implementation summary

### Development Guides
- `docs/vectorization/direction_vectorize.md` - Core vectorization principles
- `docs/FILE_ORGANIZATION.md` - Project structure guide

## References

- Original Fortran code: https://geo.phys.spbu.ru/~tsyganenko/modeling.html
- Paper: Tsyganenko, N. A. (1995), "Modeling the Earth's magnetospheric magnetic field"
- IGRF model: https://www.ngdc.noaa.gov/IAGA/vmod/igrf.html