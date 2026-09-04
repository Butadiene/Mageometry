# mageometry/viz3d/mesh.py
"""
Conversion of Mageometry objects to PyVista meshes.

Pure data construction — nothing here renders. `to_rectilinear_grid` turns a
`GriddedField` into a ``pyvista.RectilinearGrid`` with scalar quantities as
point data; `trace_polydata` turns a `FieldLineTrace` into a single
``pyvista.PolyData`` of polylines. NaN values are kept (the viewers hide
them with ``nan_opacity=0``).
"""

import numpy as np

from ..viz._quantities import QUANTITIES, resolve_quantity
from ._pv import require_pyvista

__all__ = ["to_rectilinear_grid", "trace_polydata"]

# Quantities computable directly from the stored grid values (no field
# callable, no finite differences).
_GRID_DIRECT = ('bmag', 'bx', 'by', 'bz')


def _array_name(quantity, resolved):
    """Deterministic point-data array name for a quantity."""
    if isinstance(quantity, str):
        return quantity
    for key, q in QUANTITIES.items():
        if q is resolved:
            return key
    return resolved.label


def to_rectilinear_grid(gridded_field, quantities=('bmag',), field=None,
                        delta=0.01, method='linear'):
    """
    Build a ``pyvista.RectilinearGrid`` from a `GriddedField`.

    The grid always carries the field vector as point data ``'B'``; each
    entry of ``quantities`` is evaluated on the grid nodes and attached as a
    scalar point-data array under its name.

    Parameters
    ----------
    gridded_field : GriddedField
    quantities : sequence of str, Quantity, or callable, optional
        Scalars to attach (resolved through `mageometry.viz.QUANTITIES`).
        ``'bmag'``/``'bx'``/``'by'``/``'bz'`` are computed directly from the
        stored grid values; geometry quantities (curvature, torsion, ...) are
        evaluated through a field callable and are NaN on nodes where the
        finite-difference stencil leaves the grid (the outermost layer).
    field : callable, optional
        ``field(x, y, z) -> (bx, by, bz)`` used to evaluate geometry
        quantities. Default: ``gridded_field.field(method)``. Evaluating a
        derivative quantity costs several field calls per node — coarsen
        large grids first with ``gridded_field.subvolume(stride=...)``.
    delta : float, optional
        Finite-difference step for geometry quantities.
    method : str, optional
        Interpolation method for the default field callable ('linear',
        'cubic', ...). Prefer 'cubic' for smooth derivative quantities.

    Returns
    -------
    pyvista.RectilinearGrid
        Point data: ``'B'`` (vectors) plus one scalar array per quantity.
    """
    pv = require_pyvista()
    gf = gridded_field
    grid = pv.RectilinearGrid(gf.x, gf.y, gf.z)

    # GriddedField arrays are (nx, ny, nz) with x slowest; VTK point data is
    # x-fastest, hence the Fortran-order ravel.
    grid.point_data['B'] = np.column_stack(
        [gf.b[..., k].ravel(order='F') for k in range(3)])

    nodes = None
    for quantity in quantities:
        q = resolve_quantity(quantity)
        name = _array_name(quantity, q)
        if isinstance(quantity, str) and quantity in _GRID_DIRECT:
            if quantity == 'bmag':
                vals = np.sqrt(np.sum(gf.b.astype(np.float64, copy=False) ** 2,
                                      axis=-1))
            else:
                vals = gf.b[..., 'xyz'.index(quantity[1])]
            grid.point_data[name] = np.asarray(vals, dtype=np.float64).ravel(order='F')
            continue
        if field is None:
            field = gf.field(method)
        if nodes is None:
            X, Y, Z = np.meshgrid(gf.x, gf.y, gf.z, indexing='ij')
            nodes = tuple(c.ravel(order='F') for c in (X, Y, Z))
        grid.point_data[name] = q.evaluate(field, *nodes, delta=delta)
    return grid


def trace_polydata(trace, color=None, field=None, delta=0.01, label=None):
    """
    Build a single ``pyvista.PolyData`` of polylines from a `FieldLineTrace`.

    NaN padding is trimmed; lines with fewer than 2 points are skipped (a
    one-point polyline is not a valid VTK cell). Point data always carries
    ``'s'`` (arc length from the seed, signed for ``direction='both'``) and
    ``'line_id'`` (the index of the line in the trace).

    Parameters
    ----------
    trace : FieldLineTrace
    color : quantity or ndarray, optional
        A quantity name / `Quantity` / callable (needs ``field``), or an
        array of shape ``trace.x.shape``: attached as a point-data array and
        made the active scalars.
    field : callable, optional
        Required when ``color`` is a quantity.
    delta : float, optional
        Finite-difference step for quantity evaluation.
    label : str, optional
        Array name for an ndarray ``color`` (default ``'value'``).

    Returns
    -------
    pyvista.PolyData
    """
    pv = require_pyvista()

    values = None
    name = None
    if color is not None:
        if isinstance(color, np.ndarray) and color.shape == trace.x.shape:
            values = np.asarray(color, dtype=np.float64)
            name = label or 'value'
        else:
            if field is None:
                raise ValueError("field is required to colour lines by a quantity.")
            q = resolve_quantity(color, label)
            flat = q.evaluate(field, trace.x.ravel(), trace.y.ravel(),
                              trace.z.ravel(), delta=delta)
            values = flat.reshape(trace.x.shape)
            name = _array_name(color, q)

    points, cells, s_vals, line_ids, col_vals = [], [], [], [], []
    offset = 0
    for i in range(trace.n_lines):
        n = int(trace.nsteps[i]) + 1
        if n < 2:
            continue
        x, y, z = trace.path(i)
        points.append(np.column_stack([x, y, z]))
        cells.append(np.concatenate([[n], offset + np.arange(n)]))
        s_vals.append(trace.arc_length(i))
        line_ids.append(np.full(n, i, dtype=np.int64))
        if values is not None:
            col_vals.append(values[i, :n])
        offset += n

    if not points:
        return pv.PolyData()
    poly = pv.PolyData(np.concatenate(points), lines=np.concatenate(cells))
    poly.point_data['s'] = np.concatenate(s_vals)
    poly.point_data['line_id'] = np.concatenate(line_ids)
    if values is not None:
        poly.point_data[name] = np.concatenate(col_vals)
    # Only the colour array (if any) is active: an actor added without
    # explicit scalars must not pick up 's' or 'line_id'.
    poly.point_data.active_scalars_name = name
    return poly
