# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vectorized extension of the Python geopack library for Tsyganenko magnetospheric magnetic field models (T89, T96, T01, T04) and IGRF. Provides a dual API: original scalar functions and high-performance NumPy vectorized versions (20-150x speedup). Published on PyPI as `geopack-vectorize`.

**Important:** This codebase was largely AI-generated and is being gradually human-reviewed. Treat existing code with appropriate caution.

## Build & Development Commands

```bash
# Install in development mode
pip install -e .

# Run all tests
python -m unittest discover tests/

# Run a single test file
python tests/test_vectorized_models.py

# Run with custom tolerances
GEOPACK_FIELD_RTOL=1e-8 GEOPACK_FIELD_ATOL=1e-5 python -m unittest tests/test_vectorized_models.py

# Build source distribution
python setup.py sdist

# Tag a release
git tag -a v2.0.0 -m "Release version 2.0.0"
```

Version must be updated in three places: `setup.py`, `pyproject.toml`, and `geopack/__init__.py`.

## Architecture

### Dual API Design

- **Scalar API** (`geopack.models.*`, `geopack.geopack`): Original loop-based implementations, one point at a time.
- **Vectorized API** (`geopack.vectorized.*`): NumPy-broadcasting implementations that process arrays of points simultaneously.

Both are re-exported from `geopack/__init__.py`. Vectorized functions use the `_vectorized` suffix when accessed from the top-level package (e.g., `t96_vectorized`, `trace_vectorized`).

### Package Layout

- `geopack/geopack.py` — Core scalar functions: coordinate transforms, IGRF, tracing, recalc
- `geopack/models/` — Scalar field models (t89, t96, t01, t04)
- `geopack/vectorized/models/` — Vectorized field models (same four)
- `geopack/vectorized/coordinates.py` — Vectorized coordinate transforms
- `geopack/vectorized/igrf.py` — Vectorized IGRF internal field
- `geopack/vectorized/trace.py` — Vectorized field line tracing
- `geopack/vectorized/field_line_geometry.py` — Frenet-Serret frame calculations
- `geopack/vectorized/field_line_directional_derivatives.py` — Directional derivatives along field lines
- `geopack/igrf_coeffs/` — IGRF coefficient data files (.txt)
- `tests/` — unittest-based test suite (no pytest config)
- `examples/notebooks/` — Jupyter tutorial notebooks

### Vectorization Conventions

All vectorized functions follow these rules:
1. Accept NumPy arrays for coordinate inputs (x, y, z)
2. Use `np.where()` instead of if/else for conditional logic
3. Guard against division by zero with safe division patterns
4. No global variables — all parameters passed explicitly
5. Initialize arrays with `np.zeros_like()`
6. Maintain scalar compatibility: scalar input → scalar output

### Field Model Signatures

- **T89:** `t89(iopt, ps, x, y, z)` — iopt is Kp level (1-7)
- **T96/T01/T04:** `func(parmod, ps, x, y, z)` — parmod is a 10-element array `[Pdyn, Dst, ByIMF, BzIMF, ...]`

All return `(bx, by, bz)` magnetic field components in GSM coordinates (nT).

### Testing Approach

Tests validate vectorized implementations against scalar loop results using `np.testing.assert_allclose`. Key environment variables:
- `GEOPACK_FIELD_RTOL` (default 1e-10) — relative tolerance
- `GEOPACK_FIELD_ATOL` (default 1e-6 nT) — absolute tolerance
- `GEOPACK_MAXULP` (default 32) — max ULP distance for trace tests

### Dependencies

- **Required:** numpy >= 1.16, scipy >= 1.0
- **Dev:** pytest, pytest-cov, matplotlib
- **Examples:** matplotlib, jupyter

## Language Notes

Some comments in `geopack/__init__.py` are in Japanese. The project owner prefers English for all comments.
