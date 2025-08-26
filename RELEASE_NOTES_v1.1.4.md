# Release Notes - v1.1.4

## Bug Fixes and Improvements

### Fixed Field Line Tracing Error
- Fixed critical error in `trace_field_lines_vectorized.py` that could cause tracing failures
- Improved robustness of field line tracing algorithm

### Documentation Updates
- Updated README with latest project status and improvements
- Fixed matplotlib code examples in Jupyter notebooks
- Enhanced accuracy validation notebooks with clearer visualizations
- Improved performance benchmark notebooks with updated results

### Notebook Improvements
The following notebooks have been updated with fixes and enhancements:
- `04_accuracy_validation.ipynb` - Improved accuracy validation displays
- `06_field_line_tracing_guide.ipynb` - Fixed code examples and improved clarity
- `07_field_line_tracing_performance_benchmark.ipynb` - Updated benchmarks with latest optimizations
- `08_advanced_field_line_applications.ipynb` - Enhanced application examples
- `09_field_line_tracing_path_accuracy_validation.ipynb` - Comprehensive path accuracy improvements
- `10_field_line_tracing_algorithm_validation.ipynb` - Algorithm validation updates
- `11_field_line_tracing_comprehensive_comparison.ipynb` - Updated comparison results
- `13_dipole_field_directional_derivatives.ipynb` - Improved dipole field analysis

### License Update
- Minor license file formatting improvements

## Installation

```bash
pip install --upgrade geopack-vectorized
```

## Compatibility
- Fully backward compatible with v1.1.3
- No API changes
- Python 3.6+ supported