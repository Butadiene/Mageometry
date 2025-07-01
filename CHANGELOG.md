# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.13] - 2025-01-07

### Added
- Vectorized IGRF (International Geomagnetic Reference Field) implementation
  - `igrf_geo_vectorized()` for spherical geographic coordinates
  - `igrf_gsm_vectorized()` for GSM coordinates  
  - `igrf_gsw_vectorized()` for GSW coordinates
  - 9-13x performance improvement for arrays of 1000+ points
  - Exact numerical compatibility with scalar implementation
- Vectorized coordinate transformations
  - All major coordinate systems supported (GSM, GSE, GSW, GEO, MAG, SM, GEI)
  - 25-60x speedup for batch processing
  - `coordinates_vectorized.py` and `coordinates_vectorized_complex.py` modules
- Comprehensive Jupyter notebook examples
  - 01_coordinate_transformations_guide.ipynb
  - 02_magnetic_field_models_guide.ipynb  
  - 03_performance_comparison.ipynb
  - 04_accuracy_validation.ipynb
  - 05_igrf_vectorized_guide.ipynb
- Extensive test suites for IGRF and coordinate transformations
- Performance benchmarking scripts
- Documentation for vectorization implementations

### Changed
- Updated all notebooks to use vectorized functions where applicable
- Improved documentation with detailed vectorization guides

### Fixed
- Datetime hour range issue in coordinate transformation examples
- Matplotlib compatibility issues with streamplot alpha parameter
- Memory efficiency calculation division by zero error
- Function signature handling for bspcar/bcarsp transformations
- Undefined variable in satellite orbit example

## [1.0.12] - 2024-01-30

### Added
- Fully vectorized implementations of all magnetospheric models:
  - T89 model with 50x performance improvement
  - T96 model with 30x performance improvement
  - T01 model with complete vectorization
  - T04 model with complete vectorization
- Optimized field line tracing with 265x speedup
- Vectorized coordinate transformations
- Comprehensive test suite for all vectorized models
- Performance benchmarking tools
- Example scripts and notebooks

### Changed
- Reorganized package structure with separate `models/` and `vectorized/` modules
- Updated build configuration to use pyproject.toml
- Removed platform restrictions (now supports all platforms, not just Mac OS)
- Improved documentation and examples

### Fixed
- T01 and T04 models now handle invalid X values gracefully (X < -15 Re)
- Improved numerical stability in vectorized implementations
- Fixed edge cases in coordinate transformations

## [1.0.11] - Previous releases

See git history for changes in previous versions.