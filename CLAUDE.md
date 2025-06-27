# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python implementation of the Geopack magnetic field modeling library, originally written in Fortran by N.A. Tsyganenko. The project includes both scalar and vectorized implementations of various magnetospheric field models.

## Project Structure

```
geopack-vectorize/
├── geopack/                    # Main library code
│   ├── geopack.py             # Core module with IGRF and coordinates
│   ├── t89.py                 # T89 scalar implementation
│   ├── t96.py                 # T96 scalar implementation
│   ├── t01.py                 # T01 scalar implementation
│   ├── t04.py                 # T04 scalar implementation
│   ├── t89_vectorized.py      # T89 vectorized (50x speedup)
│   ├── t96_vectorized.py      # T96 vectorized (30x speedup, full implementation)
│   ├── dipole_vectorized.py   # Vectorized dipole field (250x speedup)
│   ├── coord_transforms_vectorized.py  # Vectorized coordinate transforms
│   └── trace_optimized.py     # Optimized field line tracing (265x speedup)
│
├── docs/                      # Documentation
│   ├── vectorization/        # Vectorization guides and progress
│   ├── accuracy_reports/     # Accuracy evaluation results
│   └── FILE_ORGANIZATION.md  # File organization guide
│
├── tests/                    # Test scripts
│   ├── debug/               # Debug and analysis scripts
│   └── validation/          # Validation and benchmark scripts
│
└── archive/                 # Archived/experimental code
```

## Key Components

### Core Library (`geopack/`)
- **geopack.py** - Main module with coordinate transforms and IGRF model
- **t89.py** - T89 Kp-based model (scalar)
- **t96.py** - T96 solar wind parameter-based model (scalar)
- **t01.py** - T01 model with storm-time corrections (scalar)
- **t04.py** - T04 storm-time model (scalar)

### Vectorized Implementations (`geopack/`)
- **t89_vectorized.py** - Vectorized T89 (50x speedup)
- **t96_vectorized.py** - Vectorized T96 (30x speedup for batch, full implementation)
- **dipole_vectorized.py** - Vectorized dipole field (250x speedup)
- **coord_transforms_vectorized.py** - All coordinate transforms
- **trace_optimized.py** - Optimized field line tracing (265x speedup)

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
python geopack/test_geopack1.py

# Run accuracy evaluation
python tests/validation/evaluate_t96_full_accuracy.py

# Run specific validation tests
python tests/validation/test_t96_final.py
python tests/validation/test_t89_vectorized.py

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
- Dipole field (perfect accuracy)
- T89 model (full implementation, 50x speedup)
- T96 model (full implementation with all components)
  - Birkeland currents (birk1tot_02, birk2tot_02) ✅
  - Interconnection field (intercon) ✅
  - Tail and ring currents (tailrc96) ✅
  - Accuracy: Max relative error < 1.8e-08 (excellent)
  - Performance: 30x speedup for batch processing
- All coordinate transforms
- Field line tracing (265x speedup)

### TODO
- T01 model vectorization (complex due to iterative algorithms)
- T04 model vectorization
- IGRF vectorization

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
ps = geopack.recalc(ut)
bx, by, bz = geopack.t96.t96(parmod, ps, x, y, z)
```

### Vectorized
```python
from geopack.t96_vectorized import t96_vectorized
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