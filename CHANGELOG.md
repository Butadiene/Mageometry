# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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