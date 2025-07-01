# File Organization

## Project Structure

```
geopack-vectorize/
├── geopack/                       # Main package
│   ├── __init__.py               # Package initialization
│   ├── geopack.py                # Core functions (IGRF, coordinates)
│   ├── models/                   # Scalar model implementations
│   │   ├── __init__.py
│   │   ├── t89.py               # T89 scalar model
│   │   ├── t96.py               # T96 scalar model
│   │   ├── t01.py               # T01 scalar model
│   │   └── t04.py               # T04 scalar model
│   └── vectorized/              # Vectorized implementations
│       ├── __init__.py
│       ├── t89_vectorized.py    # T89 vectorized (50x speedup)
│       ├── t96_vectorized.py    # T96 vectorized (30x speedup)
│       ├── t01_vectorized.py    # T01 vectorized (40x speedup)
│       ├── t04_vectorized.py    # T04 vectorized (35x speedup)
│       └── condip1_exact_vectorized.py  # Vectorized dipole field
│
├── tests/                        # Test suite
│   ├── test_geopack1.py        # Original Fortran compatibility tests
│   ├── test_vectorized_models.py # Vectorized model tests
│   └── benchmark_models.py      # Performance benchmarks
│
├── examples/                     # Example code
│   ├── basic_usage.py           # Simple usage examples
│   ├── field_line_tracing.py    # Field line tracing demo
│   └── notebooks/               # Jupyter notebooks
│       ├── Field Line Trace Demo.ipynb
│       ├── field_slice_comparisons.ipynb
│       ├── t89_vectorized_evaluation.ipynb
│       ├── t96_vectorized_evaluation.ipynb
│       ├── t01_vectorized_evaluation.ipynb
│       ├── t04_vectorized_evaluation.ipynb
│       └── t96_solar_wind_evaluation.ipynb
│
├── docs/                         # Documentation
│   ├── FILE_ORGANIZATION.md     # This file
│   ├── vectorization/           # Vectorization guides
│   │   └── direction_vectorize.md
│   └── accuracy_reports/        # Accuracy evaluations
│       ├── T96_VECTORIZATION_ACCURACY_REPORT.md
│       └── T96_VECTORIZATION_SUMMARY.md
│
├── README.md                     # Project documentation
├── CLAUDE.md                     # AI assistant guidance
├── CLEANUP_SUMMARY.md           # Cleanup documentation
├── RELEASE_NOTES.md             # Release information
├── LICENSE                       # License file
├── setup.py                      # Package setup
├── pyproject.toml               # Modern Python packaging
├── MANIFEST.in                   # Distribution manifest
├── requirements.txt              # Dependencies
└── .gitignore                    # Git ignore patterns
```

## Key Files

### Core Library (`geopack/`)
- `geopack.py`: Main module with IGRF model and coordinate transforms
- `models/*.py`: Scalar implementations of T89, T96, T01, T04 models
- `vectorized/*.py`: Vectorized implementations with 20-150x speedup

### Tests (`tests/`)
- `test_geopack1.py`: Original test suite from Fortran version
- `test_vectorized_models.py`: Tests for vectorized implementations
- `benchmark_models.py`: Performance comparison tools

### Examples (`examples/`)
- `basic_usage.py`: Getting started with the library
- `field_line_tracing.py`: Demonstrates field line tracing
- `notebooks/`: Interactive Jupyter notebooks for exploration

### Documentation (`docs/`)
- `vectorization/`: Technical guides on vectorization approach
- `accuracy_reports/`: Detailed accuracy evaluations

## Import Structure

The package supports flexible imports:

```python
# Direct imports (recommended)
from geopack import t89, t96, t01, t04
from geopack import t89_vectorized, t96_vectorized, t01_vectorized, t04_vectorized

# Module imports
from geopack.models.t96 import t96
from geopack.vectorized.t96_vectorized import t96_vectorized

# Core functions
import geopack
ps = geopack.recalc(ut)
```

## Development Guidelines

1. Scalar models go in `geopack/models/`
2. Vectorized versions go in `geopack/vectorized/`
3. Tests go in `tests/`
4. Examples and notebooks go in `examples/`
5. Documentation goes in `docs/`