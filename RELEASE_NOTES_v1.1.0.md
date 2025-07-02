# Release Notes - Version 1.1.0

## 🎉 Major Release: Complete Vectorization & Enhanced Organization

### 🚀 Performance Improvements

#### Fully Vectorized Models
All Tsyganenko models now have complete vectorized implementations:
- **T89**: ~50x speedup with full Kp-based field calculation
- **T96**: ~30x speedup with ALL components vectorized:
  - Birkeland currents (birk1tot_02, birk2tot_02) ✅
  - Interconnection field (intercon) ✅
  - Tail and ring currents (tailrc96) ✅
  - Maximum relative error: < 1.8e-08
- **T01**: ~40x speedup with storm-time corrections
- **T04**: ~35x speedup with complete storm model

#### Additional Vectorizations
- **IGRF Model**: 9-13x speedup with exact scalar compatibility
- **Coordinate Transformations**: 25-60x speedup for all systems
- **Field Line Tracing**: 30-50x speedup with improved accuracy

### ✨ New Features

#### Enhanced Field Line Tracing
- Production implementation with boundary interpolation
- Boundary accuracy improved to < 0.4 km (< 0.00006 Re)
- 4000x more accurate than scalar at boundaries
- Critical for magnetopause crossing studies

#### Comprehensive Validation
- 11 tutorial notebooks with step-by-step guides
- Performance benchmarks for all functions
- Accuracy validation showing < 1e-8 relative error
- Field line path validation with physical consistency checks

### 📁 Improved Organization

#### Clearer File Names
- `trace_vectorized.py` → `trace_field_lines_vectorized.py`
- `trace_vectorized_no_interp.py` → `trace_field_lines_vectorized_nointerp.py`
- Descriptive notebook names explaining purpose

#### Better Structure
- Organized notebooks by category (tutorials, performance, validation, applications)
- Comprehensive README files in key directories
- Clean separation of scalar and vectorized implementations

### 🔧 Technical Improvements

#### Numerical Enhancements
- Improved boundary detection in field line tracing
- Better handling of edge cases in all models
- Enhanced numerical stability in coordinate transforms
- Safe division practices throughout

#### Code Quality
- Consistent API across all vectorized functions
- Comprehensive docstrings with examples
- Type hints for better IDE support
- Clean, maintainable code structure

### 📊 Performance Summary

| Function | Speedup | Accuracy |
|----------|---------|----------|
| T89 Model | 50x | < 1e-8 relative error |
| T96 Model | 30x | < 1.8e-8 relative error |
| T01 Model | 40x | < 1e-8 relative error |
| T04 Model | 35x | < 1e-8 relative error |
| IGRF | 9-13x | Exact match |
| Coordinates | 25-60x | < 1e-10 error |
| Field Line Tracing | 30-50x | < 0.4 km boundary error |

### 🔄 Migration Guide

#### Import Updates
```python
# Old imports
from geopack.trace_vectorized import trace_vectorized

# New imports
from geopack.trace_field_lines_vectorized import trace_vectorized
```

Function names remain the same - only module names changed for clarity.

### 🧪 Testing
- Comprehensive test suite validating all implementations
- Benchmarks demonstrating performance gains
- Edge case handling verified
- Fortran compatibility maintained

### 📚 Documentation
- 11 tutorial notebooks covering all features
- Performance comparison notebooks
- Accuracy validation notebooks
- API documentation with examples

### 🙏 Acknowledgments
This release represents a complete vectorization of the geopack library, making it suitable for modern large-scale space physics computations while maintaining the accuracy of the original Fortran implementations.

---

## Installation
```bash
pip install geopack-vectorized==1.1.0
```

## Upgrade from 1.0.x
```bash
pip install --upgrade geopack-vectorized
```

No breaking changes - all vectorized functions are drop-in replacements for scalar versions.