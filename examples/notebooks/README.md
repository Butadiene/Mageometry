# Mageometry Notebooks Index

Tutorial notebooks demonstrating Mageometry and its vectorized geopack field engine. Install matplotlib and jupyter to run them.

## Tutorial Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [01_coordinate_transformations_guide](01_coordinate_transformations_guide.ipynb) | Coordinate system transforms (GEI, GEO, GSM, GSE, SM, MAG, GSW) with scalar and vectorized implementations |
| 2 | [02_magnetic_field_models_guide](02_magnetic_field_models_guide.ipynb) | External Tsyganenko models (T89, T96, T01, T04), internal field models (dipole, IGRF), and total field calculation |
| 3 | [03_performance_comparison](03_performance_comparison.ipynb) | Scalar vs vectorized benchmarks across coordinates, field models, IGRF, and tracing |
| 4 | [04_accuracy_validation](04_accuracy_validation.ipynb) | Numerical accuracy verification of vectorized vs scalar implementations |
| 5 | [05_field_line_tracing_guide](05_field_line_tracing_guide.ipynb) | Field line tracing tutorial with visualization and performance comparison |
| 6 | [06_field_line_tracing_validation](06_field_line_tracing_validation.ipynb) | Tracing accuracy validation: endpoint agreement, path comparison, reversibility |
| 7 | [07_fieldline_geometry_and_derivatives](07_fieldline_geometry_and_derivatives.ipynb) | Frenet-Serret frame and the 9 directional derivative formulas |

## Advanced Examples

Detailed directional derivative map notebooks are in [`directional_derivatives_maps/`](directional_derivatives_maps/):

| Notebook | Description |
|----------|-------------|
| [dipole_field_directional_derivatives](directional_derivatives_maps/dipole_field_directional_derivatives.ipynb) | Dipole field derivative maps on meridional and equatorial planes with analytical comparison |
| [t96_field_directional_derivatives](directional_derivatives_maps/t96_field_directional_derivatives.ipynb) | T96 model derivative maps on multiple planes with field-aligned current calculations |
