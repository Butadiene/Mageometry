# mageometry/viz/__init__.py
"""
Visualization of field line geometry (requires matplotlib).

All functions take the same objects as the analysis API — a
``field(x, y, z)`` callable, a `FieldLineTrace`, coordinates — and draw on
a matplotlib axes (a new figure if ``ax`` is not given), returning the
matplotlib artists so plots can be customised further. Units are the
field's own; pass ``unit='Re'`` etc. for axis labels.

- `plot_geometry_map` — a scalar quantity (curvature, torsion, |B|, any
  directional derivative, or your own callable) on an axis-aligned plane
- `plot_field_direction` — in-plane field direction arrows
- `plot_field_lines` — traced lines, projected or in 3D, optionally coloured
  by a quantity along the line
- `plot_line_profiles` — quantities versus arc length along traced lines
- `plot_frenet_frame` — T/n/b arrows at points

Undefined values (NaN from the geometry functions) are left blank.
matplotlib is imported lazily; install it with ``pip install matplotlib``
(or ``pip install -e .[viz]``).
"""

from .maps import plot_geometry_map, plot_field_direction
from .lines import plot_field_lines, plot_line_profiles
from .frames import plot_frenet_frame
from .planes import plane_grid, plane_axes, project
from ._quantities import QUANTITIES, Quantity, resolve_quantity

__all__ = [
    "plot_geometry_map",
    "plot_field_direction",
    "plot_field_lines",
    "plot_line_profiles",
    "plot_frenet_frame",
    "plane_grid",
    "plane_axes",
    "project",
    "QUANTITIES",
    "Quantity",
    "resolve_quantity",
]
