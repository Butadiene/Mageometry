# Mageometry documentation

Mageometry analyzes magnetic field lines from a callable
`field(x, y, z) -> (bx, by, bz)`. The same analysis works with geopack
models, analytic test fields, and interpolated simulation data.

| Task | Guide |
| --- | --- |
| Compute frames, curvature, torsion, and currents | [Geometry analysis](geometry_analysis.md) |
| Load your data or write a reader | [Simulation data formats](simulation_data_formats.md) |
| Open a snapshot, select a component, and inspect slices | [Viewer guide](viewer.md) |
| Interpret rotation, shear, anisotropy, and coiling | [Transverse geometry](transverse_geometry.md) |
| Run numerical examples or a viewer from the terminal | [Runnable scripts](../README.md#runnable-scripts) |
| Follow worked numerical examples | [Notebook index](../examples/notebooks/README.md) |
| Check installation, field models, and measured performance | [Project README](../README.md) |
| Look up predecessor release history | [Release archive](releases/README.md) |

## Install for your workflow

Run these commands from the repository root. Python 3.9+ is required;
Mageometry is installed from source and is not published on PyPI.

```bash
python -m pip install -e .                 # NumPy/SciPy analysis
python -m pip install -e '.[io,viz3d]'      # HDF5/XDMF input and 3D viewers
python -m pip install -e '.[examples]'      # notebooks and benchmarks
```

The `examples` extra does not include PyVista; use `.[examples,viz3d]`
for notebooks that use 3D plotting. The `viz` extra provides Matplotlib
alone. Importing Mageometry does not download data or coefficients.

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
density, tracing, and numerical checks. For an interactive model example:

```bash
python examples/geometry_viewer.py --component gamma
```

## Start with your own snapshot

```bash
python examples/geometry_viewer_simulation.py --xmf snapshot.xmf --stride 4
```

Replace `snapshot.xmf` with your filename. The script requires an explicit
input file and has no built-in simulation path or grid size. `--stride`
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

For local verification, install `.[dev]` and run
`python -m unittest discover tests/`. The historical installation commands
and import paths in the release archive describe the predecessor project;
use the current guides above for Mageometry.
