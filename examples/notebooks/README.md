# Mageometry Notebooks Index

Tutorial notebooks for Mageometry. Install the example dependencies to run them (`pip install -e .[examples]`, or matplotlib + jupyter + pandas + h5py).

## Start here: the analysis library

| # | Notebook | Description |
|---|----------|-------------|
| 7 | [07_fieldline_geometry_and_derivatives](07_fieldline_geometry_and_derivatives.ipynb) | **Field line geometry with Mageometry**: field callables (`geopack_field`), the Frenet-Serret frame, the nine directional derivative formulas, NaN/validity conventions and the finite-difference quality diagnostic, choosing δ, geometry along traced field lines, vectorization, meridian maps |
| 8 | [08_simulation_data_geometry](08_simulation_data_geometry.ipynb) | **Simulation data pipeline**: write a compatible XDMF/HDF5 file (from a T96 reference field, so every step is checkable), `load_xdmf` / `GriddedField`, linear vs cubic interpolation, curvature through the file, tracing with `bounds`, bringing your own data |

## The geopack field engine

These cover `mageometry.geopack`, the vectorized geopack fork that serves as one field source.

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [01_coordinate_transformations_guide](01_coordinate_transformations_guide.ipynb) | Coordinate system transforms (GEI, GEO, GSM, GSE, SM, MAG, GSW) with scalar and vectorized implementations |
| 2 | [02_magnetic_field_models_guide](02_magnetic_field_models_guide.ipynb) | External Tsyganenko models (T89, T96, T01, T04), internal field models (dipole, IGRF), and total field calculation |
| 3 | [03_performance_comparison](03_performance_comparison.ipynb) | Scalar vs vectorized benchmarks across coordinates, field models, IGRF, and tracing |
| 4 | [04_accuracy_validation](04_accuracy_validation.ipynb) | Numerical accuracy verification of vectorized vs scalar implementations |
| 5 | [05_field_line_tracing_guide](05_field_line_tracing_guide.ipynb) | Engine tracer (`geopack.trace` scalar vs `trace_vectorized`), plus a closing section on the library-level `trace_field_lines` |
| 6 | [06_field_line_tracing_validation](06_field_line_tracing_validation.ipynb) | Engine tracing accuracy validation: endpoint agreement, path comparison, reversibility |

## Advanced examples

Detailed directional derivative map notebooks are in [`directional_derivatives_maps/`](directional_derivatives_maps/):

| Notebook | Description |
|----------|-------------|
| [dipole_field_directional_derivatives](directional_derivatives_maps/dipole_field_directional_derivatives.ipynb) | Dipole field derivative maps on meridional and equatorial planes with analytical comparison |
| [t96_field_directional_derivatives](directional_derivatives_maps/t96_field_directional_derivatives.ipynb) | T96 model derivative maps on multiple planes with field-aligned current calculations (stored without outputs; run it to generate the figures) |
