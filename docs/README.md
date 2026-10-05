# Mageometry documentation

Mageometry analyzes magnetic field lines from a callable
`field(x, y, z) -> (bx, by, bz)`. The same analysis works with geopack
models, analytic test fields, and interpolated simulation data.

| Task | Guide |
| --- | --- |
| Compute frames, curvature, torsion, and currents | [Geometry analysis](geometry_analysis.md) |
| Load your data or write a reader | [Simulation data formats](simulation_data_formats.md) |
| Open a snapshot, select a component, and inspect slices | [Viewer guide](viewer.md) |
| Run the unified desktop GUI and save analysis sessions | [Desktop workspace](gui.md) |
| Read Japanese explanations of every desktop control and all 22 diagnostics | [3D viewer 日本語詳細ガイド](gui_ja.md) |
| Switch datasets using shared scales, thresholds, and slices | [Dataset comparison](viewer.md#compare-datasets-in-python) |
| Develop the desktop/session code and check its boundaries | [Desktop architecture](gui_architecture.md) |
| Identify or regenerate documentation images | [Image sources and capture commands](images/README.md) |
| Interpret rotation, shear, anisotropy, and coiling | [Transverse geometry](transverse_geometry.md) |
| See how each transverse coefficient changes a flux-tube section | [Concept figure and vector downloads](transverse_decomposition_figure.md) |
| Relate directional neighbour rotation to its mean and FAC | [Directional-rotation figure and vector downloads](directional_rotation_figure.md) |
| Check baseline notation, assumptions, and R1/R2 research hypotheses | [FAC anisotropy theory](fac_anisotropy_theory.md) · [日本語訳](fac_anisotropy_theory_ja.md) |
| Run numerical examples or a viewer from the terminal | [Examples and entry points](../examples/README.md) |
| Follow worked numerical examples | [Notebook index](../examples/notebooks/README.md) |
| Check installation, field models, and measured performance | [Project README](../README.md) |
| Look up predecessor release history | [Release archive](releases/README.md) |

## Install for your workflow

Run these commands from the repository root. Python 3.9+ is required;
Mageometry is installed from source and is not published on PyPI.

```bash
python -m pip install -e .                 # NumPy/SciPy analysis
python -m pip install -e '.[io,viz3d]'      # HDF5/XDMF input and 3D viewers
python -m pip install -e '.[gui]'           # Qt desktop workspace, including file input
python -m pip install -e '.[examples]'      # notebooks and benchmarks
```

The tutorials use Matplotlib and need no PyVista. The `examples` extra
does not include PyVista; add `viz3d` for standalone viewers or your own
PyVista work. The `viz` extra provides Matplotlib alone. Importing Mageometry does not download data or coefficients.

## Start without a data file

This field has helical lines and needs only the base installation:

```python
import numpy as np
from mageometry import field_line_curvature

def field(x, y, z):
    x, y, z = np.broadcast_arrays(x, y, z)
    return -y, x, np.ones_like(z)

radius = np.array([0.5, 1.0, 2.0])
kappa = field_line_curvature(field, radius, 0.0, 0.0, delta=1e-3)
np.testing.assert_allclose(kappa, radius / (radius**2 + 1), rtol=1e-5)
print(kappa)  # approximately [0.4, 0.5, 0.4], in inverse coordinate units
```

Continue with [geometry analysis](geometry_analysis.md) for frames, current
density, tracing, and numerical checks. With `.[gui]` installed, start the
[desktop workspace](gui.md) with a model or a By comparison:

```bash
python -m mageometry.gui
python -m mageometry.gui --by -10 -5 0 5 10
```

For a standalone PyVista model viewer with `.[viz3d]` installed:

```bash
python -m mageometry.viz3d --component gamma
```

## Start with your own snapshot

In the desktop workspace, run `python -m mageometry.gui --empty`, choose
**Add model / files**, configure the reader, and **Apply and recompute**.
For the standalone CLI:

```bash
python -m mageometry.viz3d --xmf snapshot.xmf --stride 4
```

Replace `snapshot.xmf` with your filename. File mode has no built-in
simulation path or grid size. Omitting source options opens the T96 model.
The desktop example `geometry_viewer_simulation.py` requires an explicit
file or saved session. `--stride`
defaults to 1; choose a value appropriate to your grid, retaining at least
three points per axis for the viewer. See the [viewer guide](viewer.md)
for VTK, direct HDF5, custom array names, units, and memory controls.

## Reading examples and interpreting results

- Standalone examples include imports and construct their field or grid.
  File-reader recipes specify the layout and caller-supplied values they
  require; filenames and variable names are placeholders.
- Lengths and magnetic fields retain your input units. Metadata and display
  labels do not convert values.
- NaN marks undefined geometry or an invalid numerical stencil. Check
  validity per quantity; a valid tangent does not imply a valid normal.
- Finite differences and interpolation introduce their own errors. Compare
  step sizes and grid resolutions before interpreting small features.

For local verification, install `.[dev]` (or `.[dev,gui]` for Qt tests) and run
`python -m unittest discover tests/`. The historical installation commands
and import paths in the release archive describe the predecessor project;
use the current guides above for Mageometry.
