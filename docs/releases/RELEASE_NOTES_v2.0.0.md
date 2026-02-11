# Release Notes - v2.0.0

## Overview

Major release with a complete package reorganization, cleaned-up public API, full English translation, and a renamed PyPI package.

## Breaking Changes

### Package Name Changed
The PyPI package name has changed from `geopack-vectorized` to `geopack-vectorize`:
```bash
pip install geopack-vectorize
```

### Internal Module Paths Restructured
All vectorized modules have been reorganized under `geopack/vectorized/`. Users who imported from internal module paths (rather than the top-level `geopack` package) will need to update their imports:

| Old path | New path |
|---|---|
| `geopack.t89_vectorized` | `geopack.vectorized.models.t89` |
| `geopack.t96_vectorized` | `geopack.vectorized.models.t96` |
| `geopack.t01_vectorized` | `geopack.vectorized.models.t01` |
| `geopack.t04_vectorized` | `geopack.vectorized.models.t04` |
| `geopack.igrf_vectorized` | `geopack.vectorized.igrf` |
| `geopack.coordinates_vectorized` | `geopack.vectorized.coordinates` |
| `geopack.coordinates_vectorized_complex` | `geopack.vectorized.coordinates_complex` |
| `geopack.trace_field_lines_vectorized` | `geopack.vectorized.trace` |

**Top-level imports are unchanged.** All public functions remain accessible from `import geopack` with the same names (e.g., `geopack.t96_vectorized`, `geopack.igrf_geo_vectorized`).

### Removed Test and Benchmark Files
The following test/benchmark files have been removed or consolidated:
- `tests/test_igrf_vectorized.py`
- `tests/test_geopack1.py`
- `tests/test_field_line_geometry.py`
- `tests/test_field_line_directional_derivatives.py`
- `tests/test_field_line_directional_derivatives_correct.py`
- `tests/test_coordinates_vectorized.py`
- `tests/benchmark_*.py` (all benchmark scripts)

## Added

### Expanded Public API
14 additional functions are now exported from the top-level `geopack` package:
- `update_igrf`, `init_igrf`, `load_igrf` -- IGRF coefficient management
- `dip_gsw` -- dipole field in GSW coordinates
- `call_external_model`, `call_internal_model` -- generic model dispatch
- `rhand`, `step`, `trace` -- field line tracing primitives
- `shuetal_mgnp`, `t96_mgnp` -- magnetopause models

### Subpackage Namespace Access
Vectorized modules are now accessible as organized namespaces:
```python
from geopack import vectorized
from geopack.vectorized import models, coordinates, igrf, trace
```

### New Vectorized Field Line Tracing
- `geopack/vectorized/trace.py` -- rewritten vectorized tracing module
- `tests/test_trace_vectorized_with_vectorized_models.py` -- new test combining vectorized tracing with vectorized field models

### New Example Notebooks
- `05_field_line_tracing_guide.ipynb` -- field line tracing tutorial
- `06_field_line_tracing_validation.ipynb` -- tracing accuracy validation
- `07_fieldline_geometry_and_derivatives.ipynb` -- Frenet-Serret frame and directional derivatives
- `directional_derivatives_maps/t96_field_directional_derivatives.ipynb` -- T96 derivative maps
- `directional_derivatives_maps/dipole_field_directional_derivatives.ipynb` -- dipole derivative maps

### Notation Toggle in Derivative Notebooks
Added `SHOW_CHRISTOFFEL` toggle to derivative map notebooks. When `True` (default), both derivative and Christoffel symbol notations are shown. When `False`, only derivative notation is displayed. FAC panel titles now permanently use derivative notation.

## Changed

### Full English Translation
All Japanese comments, docstrings, and text throughout the codebase have been translated to English.

### NumPy-Format Docstrings
All vectorized module docstrings have been standardized to NumPy format with consistent Parameters/Returns sections.

### Standardized Examples
- Code style and comment formatting unified across all example scripts
- Markdown formatting standardized in all Jupyter notebooks
- Removed Summary sections from notebooks

### Improved Trace Testing
- `tests/test_trace_vectorized.py` significantly improved with better diagnostics

## Removed

### Deleted Example Files
- Entire `examples/conjugate_test/` directory (analysis scripts and PNG outputs)
- `examples/field_line_tracing.py`
- `examples/notebooks/00test.py`
- `examples/notebooks/validate_tracing_accuracy.py`
- 12 obsolete Jupyter notebooks (06-14 series, field analysis notebooks)

### Deleted Legacy Tracing Modules
- `geopack/trace_field_lines_vectorized.py`
- `geopack/trace_field_lines_vectorized_nointerp.py`

## Installation

```bash
pip install geopack-vectorize
```

## Compatibility
- Python 3.7+
- NumPy >= 1.16
- SciPy >= 1.0
