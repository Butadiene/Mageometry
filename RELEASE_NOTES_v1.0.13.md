# Release Notes - v1.0.13

## Release Date: January 7, 2025

We are excited to announce the release of geopack-vectorized v1.0.13, which adds comprehensive vectorized implementations of IGRF and coordinate transformations, along with extensive documentation and examples.

## Major Features

### 🚀 Vectorized IGRF Implementation

- **New Functions**: `igrf_geo_vectorized()`, `igrf_gsm_vectorized()`, `igrf_gsw_vectorized()`
- **Performance**: 9-13x speedup for arrays of 1000+ points
- **Accuracy**: Exact numerical compatibility with scalar implementation (< 1e-15 relative error)
- **Features**:
  - Handles both scalar and array inputs seamlessly
  - Proper edge case handling (poles, origin)
  - Efficient Legendre polynomial recursion
  - Processing rate: >200,000 points/second

### 🔄 Vectorized Coordinate Transformations

- **Complete Coverage**: All major coordinate systems (GSM, GSE, GSW, GEO, MAG, SM, GEI)
- **Performance**: 25-60x speedup for batch processing
- **New Modules**:
  - `coordinates_vectorized.py` - Core transformations
  - `coordinates_vectorized_complex.py` - Spherical/Cartesian conversions
- **Backward Compatible**: Scalar inputs return scalar outputs

### 📚 Comprehensive Documentation

Five new Jupyter notebooks with complete examples:

1. **01_coordinate_transformations_guide.ipynb**
   - Overview of all coordinate systems
   - Usage examples for each transformation
   - Performance comparisons
   - Visualization of coordinate relationships

2. **02_magnetic_field_models_guide.ipynb**
   - Comparison of all field models (T89, T96, T01, T04)
   - Combined IGRF + external field calculations
   - Field line visualization
   - Solar wind parameter effects

3. **03_performance_comparison.ipynb**
   - Detailed benchmarks for all vectorized functions
   - Speedup analysis vs array size
   - Memory efficiency measurements
   - Optimization recommendations

4. **04_accuracy_validation.ipynb**
   - Comprehensive accuracy validation
   - Error distribution analysis
   - Edge case testing
   - Statistical analysis of numerical precision

5. **05_igrf_vectorized_guide.ipynb**
   - IGRF usage tutorial
   - Global field maps
   - Satellite orbit analysis
   - 3D field visualization

### 🧪 Enhanced Testing

- Complete test suite for IGRF vectorization
- Coordinate transformation accuracy tests
- Performance benchmarking scripts
- Edge case validation

## Bug Fixes

- Fixed datetime hour range issue in coordinate transformation examples
- Resolved matplotlib streamplot alpha parameter compatibility
- Fixed memory efficiency calculation division by zero error
- Corrected bspcar/bcarsp function signature handling
- Fixed undefined variable in satellite orbit example

## Performance Summary

| Function Type | Typical Speedup | Processing Rate |
|--------------|-----------------|-----------------|
| IGRF | 9-13x | >200,000 pts/s |
| Coordinate Transforms | 25-60x | >1,000,000 pts/s |
| Field Models | 30-50x | >100,000 pts/s |

## Installation

```bash
pip install --upgrade geopack-vectorized
```

## Example Usage

```python
import numpy as np
import geopack
from datetime import datetime

# Initialize geopack
ut = datetime.now().timestamp()
ps = geopack.recalc(ut)

# Vectorized IGRF calculation
x = np.linspace(-10, 10, 1000)
y = np.zeros_like(x)
z = np.zeros_like(x)

# Calculate field at 1000 points simultaneously
bx, by, bz = geopack.igrf_gsm_vectorized(x, y, z)

# Vectorized coordinate transformation
x_gse, y_gse, z_gse = geopack.gsmgse_vectorized(x, y, z, 1)
```

## Acknowledgments

This release represents a significant enhancement in computational efficiency while maintaining the accuracy and reliability that users expect from geopack. We thank the community for their continued support and feedback.

## Future Plans

- GPU acceleration support
- Additional field model vectorizations
- Enhanced field line tracing algorithms
- Web-based visualization tools

For questions or issues, please visit our [GitHub repository](https://github.com/Butadiene/geopack-vectorize).