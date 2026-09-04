# mageometry/viz3d/__init__.py
"""
Interactive GPU-rendered 3D visualization (requires pyvista).

A PyVista/VTK companion to `mageometry.viz`: the same analysis objects — a
``field(x, y, z)`` callable, a `GriddedField`, a `FieldLineTrace` — in a 3D
window with free camera navigation (drag to rotate, wheel to zoom,
shift+drag to pan) and drag-able slice-plane widgets. Works the same on
Linux (including WSL2 with WSLg), Windows, and macOS.

- `slice_view` — a scalar quantity on interactive slice planes through a
  volume (three orthogonal planes, or one free plane)
- `explore` — one call: slices plus field lines traced from seed points
- `add_field_lines` — traced lines as polylines or tubes, optionally
  coloured by a quantity along the line
- `add_frenet_frame` — T/n/b arrow glyphs at points
- `to_rectilinear_grid`, `trace_polydata` — the underlying converters to
  PyVista meshes, for composing your own scenes

Quantities and colour scales follow the `mageometry.viz` conventions:
names resolved through `mageometry.viz.QUANTITIES` (or a callable), log
colour scale for positive quantities, symmetric diverging for signed ones,
and NaN (undefined) stays blank. Viewer functions accept an existing
``plotter``; ``show=False`` returns it un-shown for further composition.

pyvista is imported lazily; install it with ``pip install pyvista`` (or
``pip install -e .[viz3d]``).
"""

from .mesh import to_rectilinear_grid, trace_polydata
from .slicer import slice_view
from .lines import add_field_lines
from .frames import add_frenet_frame
from .explore import explore

__all__ = [
    "slice_view",
    "explore",
    "add_field_lines",
    "add_frenet_frame",
    "to_rectilinear_grid",
    "trace_polydata",
]
