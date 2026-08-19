# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Mageometry** is a magnetic field line geometry analysis library: Frenet-Serret frames, curvature, torsion, and directional derivatives along field lines. The analysis toolkit (`mageometry.geometry`) is the primary product; the vectorized geopack implementation (`mageometry.geopack`, Tsyganenko models T89/T96/T01/T04, IGRF, field line tracing) serves as one magnetic field source. Planned additions: reading magnetic fields from simulation output files (`mageometry.io`) and visualization (`mageometry.viz`).

**Project status — read this first:**

- This is a **hard fork** of `geopack-vectorize` (2026-08-19) and is **not published on PyPI**. The `pyproject.toml` carries the `Private :: Do Not Upload` classifier; never remove it or restore PyPI publishing metadata/workflows.
- **Breaking changes are explicitly allowed.** There is no obligation to preserve backward compatibility with geopack-vectorize. Do not add new compatibility shims or deprecation cycles; existing legacy aliases (e.g. top-level `*_vectorized` names for Mageometry functions) may be removed when convenient.
- **Provenance:** most of the inherited code was generated a year ago by a much less capable AI model and has only been partially human-reviewed. Treat existing code, comments, and docs with skepticism — verify claims (accuracy numbers, API descriptions) against the code itself before relying on them, and feel free to restructure aggressively.

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

# Build a distribution (rarely needed; the package is not published)
python -m build
```

There is no `setup.py`/`setup.cfg` — all packaging lives in `pyproject.toml`. The version must be kept in sync in two places: `pyproject.toml` and `mageometry/__init__.py` (`__version__`). Current line: `0.1.0.dev0`.

## Architecture

### API Design

- **Geometry API** (`mageometry.geometry`, re-exported at top level): Field line geometry analysis — the primary product and the part of the codebase under active development. `from mageometry import field_line_curvature` etc. All geometry functions take the magnetic field as a generic callable ``field(x, y, z) -> (bx, by, bz)`` (GSM Re in, nT out) — they have no knowledge of geopack signatures.
- **Field adapters** (`mageometry.fields`, re-exported at top level): Builders that produce such callables. `geopack_field(external, internal, parmod, ps)` wraps the geopack models (external: t89/t96/t01/t04 or None, internal: dip/igrf or None) with parameters bound at construction. Future simulation-data fields plug in here.
- **Field engine** (`mageometry.geopack`): The vectorized geopack fork, providing magnetic field models as one field source. Contains both APIs of the original project:
  - **Scalar API** (`mageometry.geopack.models.*`, `mageometry.geopack.geopack`): Original loop-based implementations, one point at a time. Kept mainly as the validation reference for the vectorized code.
  - **Vectorized API** (`mageometry.geopack.vectorized.*`): NumPy-broadcasting implementations that process arrays of points simultaneously (20-150x speedup). Vectorized functions use the `_vectorized` suffix when accessed from `mageometry.geopack` (e.g., `t96_vectorized`, `trace_vectorized`).
- **Simulation data input** (`mageometry.io`): `GriddedField` wraps a rectilinear grid + B components and builds interpolating field callables (`.field(method=...)`, scipy `RegularGridInterpolator`). File-format readers are thin adapters whose only job is to construct a `GriddedField` — currently `load_xdmf` (XDMF/HDF5 uniform grids) and `load_hdf5`; add new formats the same way. HDF5 access needs the optional `h5py` dependency (lazy import). Units are the data's own grid units throughout.
- **Planned**: `mageometry.viz` (visualization). Not yet implemented.

There is no top-level `geopack` package anymore — this also avoids shadowing the upstream `geopack` PyPI package.

### Package Layout

- `mageometry/__init__.py` — Version, geometry API re-exports, `geopack` subpackage
- `mageometry/fields.py` — Field-source adapters (`geopack_field`)
- `mageometry/io/gridded_field.py` — `GriddedField`: gridded data + interpolation
- `mageometry/io/xdmf.py` — XDMF and HDF5 readers (`load_xdmf`, `load_hdf5`)
- `mageometry/geometry/field_line_geometry.py` — Frenet-Serret frame calculations
- `mageometry/geometry/field_line_directional_derivatives.py` — Directional derivatives along field lines
- `mageometry/geopack/geopack.py` — Core scalar functions: coordinate transforms, IGRF, tracing, recalc
- `mageometry/geopack/core.py` — Currently an empty placeholder (reserved for restructuring)
- `mageometry/geopack/models/` — Scalar field models (t89, t96, t01, t04)
- `mageometry/geopack/vectorized/models/` — Vectorized field models (same four)
- `mageometry/geopack/vectorized/coordinates.py`, `coordinates_complex.py` — Vectorized coordinate transforms
- `mageometry/geopack/vectorized/igrf.py` — Vectorized IGRF internal field
- `mageometry/geopack/vectorized/trace.py` — Vectorized field line tracing
- `mageometry/geopack/igrf_coeffs/` — IGRF coefficient data files (.txt)
- `tests/` — unittest-based test suite (no pytest config)
- `benchmark/` — Scripts that regenerate the README performance/validation tables
- `examples/notebooks/` — Jupyter tutorial notebooks
- `docs/simulation_data_formats.md` — Accepted `mageometry.io` data formats and how to write compatible files (keep in sync with reader changes)
- `docs/releases/` — Historical release notes from the geopack-vectorize era (kept as record only)

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
- **Optional (`[io]` extra):** h5py >= 3.0 — only needed for `load_xdmf`/`load_hdf5`
- **Dev:** pytest, pytest-cov, matplotlib, h5py
- **Examples:** matplotlib, jupyter, pandas, psutil

## Language Notes

All comments, docstrings, and documentation must be in English. If you encounter leftover Japanese comments, translate them.
