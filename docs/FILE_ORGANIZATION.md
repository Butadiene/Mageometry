# File Organization

## Directory Structure

```
geopack-vectorize/
├── geopack/                    # Main library code
│   ├── geopack.py             # Core module with IGRF and coordinates
│   ├── t89.py                 # T89 scalar implementation
│   ├── t89_vectorized.py      # T89 vectorized implementation
│   ├── t96.py                 # T96 scalar implementation
│   ├── t96_vectorized.py      # T96 vectorized implementation (complete)
│   ├── t01.py                 # T01 scalar implementation
│   ├── t01_full_vectorized.py # T01 vectorized implementation
│   ├── t04.py                 # T04 scalar implementation
│   ├── dipole_vectorized.py   # Vectorized dipole field
│   ├── coord_transforms_vectorized.py  # Vectorized coordinate transforms
│   ├── trace_optimized.py     # Optimized field line tracing
│   └── test_geopack1.py       # Original test suite
│
├── docs/                       # Documentation
│   ├── vectorization/         # Vectorization guides and progress
│   │   ├── direction_vectorize.md
│   │   ├── direction_vectorize_2.md
│   │   ├── direction_vectorize_3.md
│   │   ├── direction_vectorized_4.md
│   │   ├── direction_vectorized_5.md
│   │   ├── VECTORIZATION_SUMMARY.md
│   │   ├── EXACT_VECTORIZATION_PROGRESS.md
│   │   ├── T96_EXACT_VECTORIZATION_SUMMARY.md
│   │   └── BIRK2_VECTORIZATION_STATUS.md
│   │
│   ├── accuracy_reports/      # Accuracy evaluation results
│   │   ├── T96_VECTORIZATION_ACCURACY_REPORT.md
│   │   ├── T96_VECTORIZATION_SUMMARY.md
│   │   ├── T96_ACCURACY_EVALUATION_REPORT.md
│   │   ├── T96_ACCURACY_ISSUES.md
│   │   ├── T96_VECTORIZATION_ANALYSIS.md
│   │   ├── T96_VECTORIZATION_COMPLETE.md
│   │   ├── T96_VECTORIZATION_FINAL_ACCURACY.md
│   │   └── t96_worst_cases.txt
│   │
│   └── FILE_ORGANIZATION.md   # This file
│
├── tests/                     # Test scripts
│   ├── debug/                # Debug and analysis scripts
│   │   ├── debug_*.py       # Various debugging scripts
│   │   ├── analyze_t96_precision.py
│   │   └── check_scalar_d.py
│   │
│   └── validation/          # Validation and benchmark scripts
│       ├── test_*.py       # Component and integration tests
│       ├── evaluate_t96_*.py  # Accuracy evaluation scripts
│       └── benchmark_t96_final.py
│
├── archive/                  # Archived/experimental code
│   ├── condip1_basis_vectorized.py
│   ├── condip1_exact_vectorized.py
│   └── t96_vectorized_exact.py
│
├── CLAUDE.md                # Instructions for Claude Code
├── README.md                # Project README
└── setup.py                 # Package setup script
```

## Key Files

### Production Code
- `geopack/t96_vectorized.py` - Complete vectorized T96 with 30x speedup
- `geopack/t89_vectorized.py` - Vectorized T89 with 50x speedup
- `geopack/t01_full_vectorized.py` - Vectorized T01 with 2384x speedup

### Documentation
- `CLAUDE.md` - Development guidelines and project overview
- `docs/accuracy_reports/T96_VECTORIZATION_ACCURACY_REPORT.md` - Final accuracy analysis
- `docs/vectorization/direction_vectorize.md` - Vectorization principles

### Testing
- `tests/validation/evaluate_t96_full_accuracy.py` - Comprehensive accuracy evaluation
- `geopack/test_geopack1.py` - Original test suite

## Cleanup Summary

### Files Organized
- 5 vectorization guides → `docs/vectorization/`
- 8 accuracy reports → `docs/accuracy_reports/`
- 30+ debug scripts → `tests/debug/`
- 40+ test scripts → `tests/validation/`
- 3 experimental files → `archive/`

### Root Directory
Now contains only essential files:
- `CLAUDE.md` - Development instructions
- `README.md` - Project documentation
- `setup.py` - Package configuration