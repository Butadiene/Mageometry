# Notebook tutorials

[Examples](../README.md) · [Analysis API](../../docs/geometry_analysis.md)
· [Theory](../../docs/fac_anisotropy_theory.md) · [理論の日本語訳](../../docs/fac_anisotropy_theory_ja.md)

Install from the repository root with `python -m pip install -e '.[examples]'`.
Open a notebook in Jupyter and run it top to bottom in a fresh kernel. Every
notebook is self-contained: small deterministic examples, no external data or
network access, and no Qt/PyVista requirement. They use the installed package;
there is no working-directory change or `sys.path` injection.

## Reading order

| Notebook | What it demonstrates |
| --- | --- |
| [01 — Coordinates](01_coordinate_transformations_guide.ipynb) | Explicit UTC epoch, forward/inverse rotations, scalar/array agreement and vector components |
| [02 — Magnetic models](02_magnetic_field_models_guide.ipynb) | T89/T96/T01/T04 callables, shared source adapters, domain masking, model comparisons and geometry |
| [03 — Tracing](03_field_line_tracing_guide.ipynb) | Generic tracing, termination codes and the opposite geopack direction convention |
| [04 — Frames and derivatives](04_fieldline_geometry_and_derivatives.ipynb) | Analytic helix, nine independent projections, NaN masks, step convergence, dipole maps |
| [05 — Sampled data](05_simulation_data_geometry.ipynb) | Temporary XDMF/HDF5 round trip, axis order, linear/cubic interpolation errors, bounded tracing |
| [06 — Python plotting](06_visualization.ipynb) | Matplotlib maps, paths, profiles, frame arrows and explicit transverse diagnostic callables |
| [07 — Current and along-field derivatives](07_current_density_from_geometry.ipynb) | Known Cartesian curl, current conversion, dipole cancellation, analytic `dalpha_ds`/`dalpha_ds_over_B`/`dfac_ds` |
| [08 — Transverse geometry and FAC](08_transverse_geometry_and_fac.ipynb) | Equal FAC with different shear, directional rotation, total-frame background attribution and nonadditive norms |

Start at 01 for model-based workflows, or at 04 for geometry with a field
callable you already have. The [desktop guide](../../docs/gui.md) and
[standalone viewer guide](../../docs/viewer.md) cover interactive applications;
notebooks focus on analysis from Python.

## Reproduce the tutorials

Tracked notebooks contain source and assertions, without stale outputs or
execution counts. Execute all eight in independent kernels, retaining results
outside the source tree for review:

```bash
python benchmark/check_notebooks.py --output /tmp/mageometry-notebooks
# Or rerun one notebook:
python benchmark/check_notebooks.py examples/notebooks/07_current_density_from_geometry.ipynb \
  --output /tmp/mageometry-notebooks
```

The runner executes every cell with the normal notebook parameters, validates
notebook structure, fails on errors, and never overwrites input notebooks.
The file tutorial cleans up its generated data; use its documented reader
contract when substituting your own snapshots. Assertions check these examples,
not universal tolerances for observations or simulations.

## Consolidated material

The former validation/benchmark notebooks duplicated maintained test and
benchmark code. The historical filenames below predate the current 01–08
numbering; their replacements are:

| Retired material | Current location |
| --- | --- |
| `03_performance_comparison` | `benchmark/readme_benchmarks.py` and `benchmark/readme_overhead_decomposition.py`; [measured results](../../README.md#performance-benchmarks) |
| `04_accuracy_validation` | `tests/test_vectorized_models.py` and `benchmark/readme_validation.py`; small scalar/array checks in 01/02 |
| `06_field_line_tracing_validation` | `tests/test_trace_vectorized.py`, `tests/test_trace_vectorized_with_vectorized_models.py`, `tests/test_tracing.py`; tutorial checks in 03 |
| Separate dipole/T96 directional-map notebooks | One callable-based map workflow in 04; source adapters in 02, current diagnostics in 07, transverse diagnostics in 08 |

Run the full regression suite with `python -m unittest discover tests/`.
Benchmark scripts report measured performance independently of tutorial execution.
