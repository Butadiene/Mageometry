# Geopack Notebooks Index

This directory contains Jupyter notebooks demonstrating various aspects of the geopack library.

## 📚 Tutorials (Start Here)
- `01_coordinate_transformations_guide.ipynb` - Learn about coordinate systems (GSM, GSE, MAG, etc.)
- `02_magnetic_field_models_guide.ipynb` - Introduction to T89, T96, T01, T04 models
- `05_igrf_vectorized_guide.ipynb` - Using the IGRF internal field model

## 🚀 Performance Benchmarks
- `03_performance_comparison.ipynb` - Overall vectorization performance gains
- `07_field_line_tracing_performance_benchmark.ipynb` - Field line tracing speed analysis

## ✅ Accuracy Validation
- `04_accuracy_validation.ipynb` - General accuracy validation of vectorized implementations
- `09_field_line_tracing_path_accuracy_validation.ipynb` - Detailed path-level accuracy of field line tracing
- `10_field_line_tracing_algorithm_validation.ipynb` - Validates tracing algorithm correctness
- `11_field_line_tracing_comprehensive_comparison.ipynb` - Compares both vectorized implementations

### Model-Specific Validations
- `t89_vectorized_evaluation.ipynb` - T89 model validation
- `t96_vectorized_evaluation.ipynb` - T96 model validation  
- `t01_vectorized_evaluation.ipynb` - T01 model validation
- `t04_vectorized_evaluation.ipynb` - T04 model validation

## 🔬 Applications
- `t96_solar_wind_evaluation.ipynb` - Analyzing field behavior under different solar wind conditions
- `field_slice_comparisons.ipynb` - Visualizing magnetic field patterns

## Key Implementation Files

### Field Line Tracing (in `geopack/`)
- `trace_field_lines_vectorized.py` - **Production implementation** with boundary interpolation
  - Use this for all scientific applications
  - Accurate boundary intersections (< 0.4 km error)
  - 30-50x speedup for batch processing
  
- `trace_field_lines_vectorized_nointerp.py` - **Validation-only implementation**
  - Matches scalar boundary behavior exactly
  - Only for testing/validation purposes
  - Less accurate at boundaries (~1500 km error)
