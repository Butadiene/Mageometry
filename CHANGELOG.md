# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking
- **Package inverted: the analysis library is now the top-level package.** The
  import root changed from `geopack` to `mageometry`:
  - Field line geometry (formerly `geopack.Mageometry`) now lives in
    `mageometry.geometry` and is re-exported at the top level
    (`from mageometry import field_line_curvature`).
  - The vectorized geopack fork moved to `mageometry.geopack`
    (`from mageometry import geopack; geopack.recalc(ut)`), acting as one
    magnetic field source for the analysis library. There is no top-level
    `geopack` package anymore, so the upstream `geopack` PyPI package is no
    longer shadowed.
  - Planned subpackages: `mageometry.io` (simulation output readers) and
    `mageometry.viz` (visualization).
- **Geometry functions now take a generic field callable.** The signature
  changed from `func(model_func, parmod, ps, x, y, z, ...)` to
  `func(field, x, y, z, ...)`, where ``field(x, y, z) -> (bx, by, bz)`` is any
  callable (GSM Re in, nT out). This decouples the analysis library from
  geopack-specific parameters and lets simulation-data fields plug in directly.

- **Geometry validity conventions reworked.** Undefined or unreliable
  quantities are now **NaN** instead of zero: the tangent where |B| is zero or
  non-finite (magnetic nulls, points outside an interpolated grid), the normal
  and binormal where curvature vanishes or the finite difference is
  unresolved, and every directional derivative at points whose stencil is
  invalid. Mask results with `np.isfinite`. The unit-dependent absolute cutoff
  `|B| > 1e-10` is gone (only zero/non-finite fields are undefined), and the
  hard-coded orthogonality cutoff `cos_theta < 1e-3` became the keyword
  `orthogonality_tol` (default 0.1) on `field_line_normal`, `_binormal`,
  `_torsion`, `_frenet_frame`, `_geometry_complete`, and
  `field_line_directional_derivatives` (which also gained `normal_flip_tol`,
  default 0.9, for the previously hidden n(+δ)·n(−δ) check). The old 1e-3
  cutoff scaled as δ², so at δ = 0.25 (a typical grid cell) it silently zeroed
  the normal at 40–60 % of points. `verify_unit_vectors` lost its unused `tol`
  argument.
- **Principal normal is now the component of dT/ds perpendicular to T**
  (Gram–Schmidt projection) rather than the raw finite difference. The frame
  is orthonormal to round-off by construction at any `delta`; curvature is
  |(dT/ds)_⊥| (differs from the raw magnitude by O(cos_theta²), below the
  truncation error). Antisymmetry residuals of the tangential and normal
  relations drop from ~1e-5 to ~1e-14; other values change by ≤1e-5.
- Geometry internals restructured around a single `_frame` pass (three field
  evaluations): `field_line_torsion` now costs 9 field evaluations instead of
  16 and `field_line_directional_derivatives` 21 instead of 77.

### Added
- **`mageometry.viz`: visualization** (optional matplotlib dependency, `[viz]`
  extra; `from mageometry import viz`). Functions draw the analysis objects
  directly and return matplotlib artists: `plot_geometry_map` (any named
  quantity — curvature, torsion, frame quality, |B|, components, every
  directional-derivative key — or a `quantity(field, x, y, z)` callable on an
  axis-aligned plane, with optional field-direction arrows and a mask),
  `plot_field_direction`, `plot_field_lines` (a `FieldLineTrace` projected onto a
  plane or on a 3D axes, plain or coloured by a quantity along the path),
  `plot_line_profiles` (quantities versus arc length), and `plot_frenet_frame`
  (T/n/b arrows, 2D or 3D). Colour scales follow each quantity's convention
  (log for positive quantities, symmetric diverging for signed ones); NaN is
  left blank. `viz.plane_grid` / `viz.project` expose the plane sampling for
  custom plots. New notebook `09_visualization`; `tests/test_viz.py`.
- **Bring-your-own-data support.** `docs/simulation_data_formats.md` is now a
  hands-on guide for adapting arbitrary MHD output to `GriddedField`: raw
  binaries, Fortran unformatted dumps, per-rank chunk files, VTK/NetCDF/HDF5
  layouts, cell-centered/staggered grids, non-uniform axes, units and
  coordinate conventions, per-step series, plus a validation checklist and a
  reader template. Supporting library pieces: `mageometry.io.read_fortran_records`
  / `iter_fortran_records` (Fortran sequential files, any byte order, 4- or
  8-byte markers), `FieldSeries` (lazy time series from any per-step loader via
  `FieldSeries.from_files(paths, loader, times)`; `XdmfSeries` is now its XDMF
  flavour), and `GriddedField.divergence()` (relative |∇·B| h/|B|: ~1e-3 for
  correctly assembled data, ~0.1 for transposed axes or permuted/sign-flipped
  components).
- **geopack no longer touches the network on import.** `init_igrf()` loads the
  bundled IGRF coefficient files only; the old import-time HTTP check against
  NOAA (which could stall imports indefinitely without connectivity) is gone.
  `update_igrf(local_dir=None, timeout=30)` is an explicit opt-in download that
  returns the files it fetched; the "Load IGRF coefficients ..." import message
  is gone too.
- **`mageometry.io` extensions.** `load_xdmf` / `load_hdf5` take
  `region=((xmin, xmax), (ymin, ymax), (zmin, zmax))` and `stride` to read a
  sub-box or coarsened grid as an HDF5 hyperslab (the full array never enters
  memory); `GriddedField.subvolume(region, stride)` and the helper
  `mageometry.io.region_slices` do the same in memory. Cell-centered XDMF
  attributes (`Center="Cell"`) are accepted and placed at the cell centers
  (`metadata['center']` records the convention; mixed centering is rejected);
  a grid `<Time>` lands in `metadata['time']`. New `load_xdmf_series(path)`
  opens time series lazily as an `XdmfSeries` (`times`, `series[i]`,
  `series.at(t)`, iteration, slicing) from either an XDMF temporal collection
  or a ParaView `.xmf.series` JSON index; reader options apply per step.
  Readers no longer make an extra native-endian copy of each component
  (`GriddedField` casts big-endian data while stacking).
- Notebooks brought in line with the library-first design: `07` rewritten as
  the geometry tutorial (field callables, NaN conventions, `field_line_frame_quality`,
  choosing δ, geometry along `trace_field_lines` paths); new `08_simulation_data_geometry`
  walking the `mageometry.io` pipeline on a reproducible T96-derived file; `05` gains a
  `trace_field_lines` section and points to it as the analysis-level tracer; the T96
  derivative-map notebook uses `geopack_field` instead of hand-written model wrappers.
  All executed notebooks have refreshed outputs.
- `field_line_frame_quality(field, x, y, z, delta)`: the finite-difference
  consistency diagnostic `cos_theta = |T·dT/ds| / |dT/ds|` (~δ²|κ'|/3κ), for
  choosing `delta` and `orthogonality_tol`. `DEFAULT_ORTHOGONALITY_TOL` and
  `DEFAULT_NORMAL_FLIP_TOL` are exported from `mageometry.geometry`.
- `tests/test_geometry_validity.py`: NaN conventions, orthonormality by
  construction, tolerance behaviour, δ² scaling of the quality diagnostic,
  interpolated-domain edges.
- `mageometry.fields.geopack_field(external, internal, parmod, ps)` (also
  exported at top level): builds a ``field(x, y, z)`` callable from the geopack
  models — external 't89'/'t96'/'t01'/'t04' or None, internal 'dip'/'igrf' or
  None — with model parameters bound at construction time.
- `tests/test_field_line_geometry.py`: first test coverage for the geometry
  API, including an analytic dipole check (equatorial curvature κ = 3/r),
  Frenet frame orthonormality, antisymmetry relations, and adapter behavior.
- **`mageometry.io`: magnetic fields from simulation output files.**
  `GriddedField` holds a rectilinear grid + B components and builds
  interpolating ``field(x, y, z)`` callables for the geometry API; readers are
  thin adapters constructing a `GriddedField`. Provided readers: `load_xdmf`
  (XDMF-described uniform grids with HDF5 heavy data) and `load_hdf5` (plain
  HDF5 datasets). Requires the new optional `h5py` dependency (`[io]` extra).
  Validated end-to-end against a real magnetosphere MHD run (local data, not
  part of this repository): the inner dipole region reproduces the analytic
  equatorial curvature 3/r. See
  `examples/python_code_samples/mhd_gridded_field_example.py` and
  `tests/test_io_gridded_field.py`.
- **`mageometry.tracing.trace_field_lines`** (also exported at top level): field
  line tracing for any ``field(x, y, z)`` callable, so Tsyganenko fields and
  simulation-data fields are traced by the same code. Uses geopack's RK5 scheme
  with per-step halving error control, but with generic termination: inner /
  outer spheres (crossings placed exactly on the sphere), a bounding box (e.g.
  `GriddedField.bounds`), a user `stop(x, y, z)` callable, `max_steps`, and
  automatic termination where the field is undefined (NaN outside an
  interpolated grid, magnetic nulls). `direction=1/-1/'both'`; results come
  back as a `FieldLineTrace` with NaN-padded paths, arc length from the seed,
  step counts, and status codes. Validated against the analytic dipole
  (L-shell invariant ~1e-7, footpoint latitude, curvature along the traced
  line) and against `geopack.trace_vectorized` on T89+IGRF (footpoints agree
  within geopack's own ~1e-2 Re boundary overshoot). The geopack port
  `geopack.trace_vectorized` is unchanged and stays as the scalar-faithful
  reference.
- Tsyganenko file round-trip test (`TestTsyganenkoFileRoundtrip`): a
  T96+dipole field is sampled on a grid, written as an XDMF/HDF5 pair in the
  MHD on-disk convention, loaded back, and compared against direct model
  evaluation — field values (<1e-2 rel), curvature through the file (2e-2
  linear / 5e-3 cubic), and Frenet frames including the validity mask. This
  pins the whole file pipeline to an exactly reproducible reference.
- **Project forked** from `geopack-vectorize` as **Mageometry** (2026-08-19). The
  package is no longer published on PyPI; the distribution name changed to
  `mageometry` and carries the `Private :: Do Not Upload` classifier to block
  accidental uploads. Install from source only.
- Version reset to `0.1.0.dev0`. Breaking changes may land without deprecation
  cycles; no backward compatibility with geopack-vectorize releases is guaranteed.
- Removed `setup.py` / `setup.cfg` (packaging consolidated into `pyproject.toml`),
  `CITATION.cff`, and the PyPI release guides under `docs/`.
- Removed the legacy top-level `*_vectorized` aliases for the field line geometry
  functions (`field_line_tangent_vectorized`, `field_line_curvature_vectorized`,
  `field_line_normal_vectorized`, `field_line_binormal_vectorized`,
  `field_line_torsion_vectorized`, `field_line_frenet_frame_vectorized`,
  `field_line_geometry_complete_vectorized`,
  `field_line_directional_derivatives_vectorized`). Use the plain names
  (e.g. `geopack.field_line_curvature`) or import from `geopack.Mageometry`.

## [2.0.0] - 2026-02-11

### Breaking
- **Package renamed** from `geopack-vectorized` to `geopack-vectorize` on PyPI
- Reorganized all vectorized modules into `geopack/vectorized/` subpackage
  - Field models moved to `geopack/vectorized/models/` (t89, t96, t01, t04)
  - Coordinates, IGRF, tracing, geometry, derivatives moved under `geopack/vectorized/`
  - Removed `_vectorized` suffix from internal filenames
- Removed legacy tracing modules (`trace_field_lines_vectorized.py`, `trace_field_lines_vectorized_nointerp.py`)
- Removed several test/benchmark files (consolidated into remaining test suite)

### Added
- 14 new public functions exported from top-level `geopack` package
- Subpackage namespace access (`from geopack.vectorized import models, coordinates, igrf, trace`)
- Rewritten vectorized field line tracing (`geopack/vectorized/trace.py`)
- New test file for trace + vectorized model integration
- 5 new Jupyter notebooks (tracing guide, tracing validation, geometry/derivatives, T96 and dipole derivative maps)
- `SHOW_CHRISTOFFEL` notation toggle in derivative map notebooks

### Changed
- Translated all Japanese comments and docstrings to English
- Standardized all docstrings to NumPy format
- Unified code style and markdown formatting across examples and notebooks
- Improved trace test diagnostics

### Removed
- Entire `examples/conjugate_test/` directory
- 12 obsolete Jupyter notebooks
- Legacy example scripts and analysis outputs

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