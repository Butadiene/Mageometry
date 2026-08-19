# mageometry/io/gridded_field.py
"""
Gridded magnetic field data and interpolation.

`GriddedField` is the common landing point for all file-format readers: a
reader's only job is to produce three coordinate axes and the magnetic field
components on that grid, and construct a `GriddedField` from them. The class
then builds ``field(x, y, z) -> (bx, by, bz)`` callables (via
`scipy.interpolate.RegularGridInterpolator`) that plug directly into the
geometry functions in `mageometry.geometry`.

Units are the data's own: positions are in the grid's length unit and field
values in the grid's field unit. Geometry results are correspondingly in
1/length-unit (curvature, torsion, directional derivatives). Rescale the axes
or field arrays before construction if you need specific units.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class GriddedField:
    """
    A magnetic field sampled on a rectilinear grid.

    Parameters
    ----------
    x, y, z : array_like, 1D
        Strictly monotonic increasing coordinate axes.
    bx, by, bz : array_like
        Field components with shape ``(len(x), len(y), len(z))``.
    metadata : dict, optional
        Free-form provenance information (source file, variable names, ...).

    Notes
    -----
    The three components are stored stacked in a single
    ``(len(x), len(y), len(z), 3)`` array; `bx`/`by`/`bz` are views into it,
    so no memory is duplicated. Large float32 inputs are kept as float32.
    """

    def __init__(self, x, y, z, bx, by, bz, metadata=None):
        self.x = np.ascontiguousarray(x, dtype=np.float64)
        self.y = np.ascontiguousarray(y, dtype=np.float64)
        self.z = np.ascontiguousarray(z, dtype=np.float64)
        for name, axis in (('x', self.x), ('y', self.y), ('z', self.z)):
            if axis.ndim != 1 or axis.size < 2:
                raise ValueError(f"Axis {name!r} must be 1D with at least 2 points.")
            if not np.all(np.diff(axis) > 0):
                raise ValueError(f"Axis {name!r} must be strictly increasing.")

        shape = (self.x.size, self.y.size, self.z.size)
        dtype = np.result_type(np.asarray(bx).dtype, np.float32)
        self.b = np.empty(shape + (3,), dtype=dtype)
        for i, (name, comp) in enumerate((('bx', bx), ('by', by), ('bz', bz))):
            comp = np.asarray(comp)
            if comp.shape != shape:
                raise ValueError(
                    f"{name} has shape {comp.shape}, expected {shape} "
                    "(= (len(x), len(y), len(z)))."
                )
            self.b[..., i] = comp
        self.metadata = dict(metadata) if metadata else {}

    @property
    def bx(self):
        return self.b[..., 0]

    @property
    def by(self):
        return self.b[..., 1]

    @property
    def bz(self):
        return self.b[..., 2]

    @property
    def shape(self):
        return self.b.shape[:3]

    @property
    def bounds(self):
        """((xmin, xmax), (ymin, ymax), (zmin, zmax)) of the grid."""
        return ((self.x[0], self.x[-1]),
                (self.y[0], self.y[-1]),
                (self.z[0], self.z[-1]))

    def __repr__(self):
        (x0, x1), (y0, y1), (z0, z1) = self.bounds
        return (f"GriddedField(shape={self.shape}, "
                f"x=[{x0:g}, {x1:g}], y=[{y0:g}, {y1:g}], z=[{z0:g}, {z1:g}])")

    def field(self, method='linear', fill_value=np.nan):
        """
        Build a ``field(x, y, z) -> (bx, by, bz)`` interpolating callable.

        The returned callable follows the Mageometry field convention: it
        accepts scalars or NumPy arrays and returns scalars for scalar input.
        Pass it directly to the `mageometry.geometry` functions.

        Parameters
        ----------
        method : str, optional
            Interpolation method understood by `RegularGridInterpolator`
            ('linear', 'nearest', 'cubic', 'quintic', ...). Default 'linear'.
            Note that finite-difference geometry quantities (curvature etc.)
            are noisy under 'linear' interpolation because its derivative is
            piecewise constant; prefer 'cubic' for geometry analysis if the
            grid fits in memory budget.
        fill_value : float or None, optional
            Value returned outside the grid bounds (default NaN). Use None to
            raise an error on out-of-bounds points instead.

        Returns
        -------
        field : callable
            ``field(x, y, z) -> (bx, by, bz)``.
        """
        bounds_error = fill_value is None
        interp = RegularGridInterpolator(
            (self.x, self.y, self.z), self.b, method=method,
            bounds_error=bounds_error,
            fill_value=None if bounds_error else fill_value,
        )

        def field(x, y, z):
            scalar_input = np.isscalar(x) and np.isscalar(y) and np.isscalar(z)
            x_arr, y_arr, z_arr = np.broadcast_arrays(
                np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(z)
            )
            pts = np.column_stack([x_arr.ravel(), y_arr.ravel(), z_arr.ravel()])
            b = interp(pts).reshape(x_arr.shape + (3,))
            bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
            if scalar_input:
                return bx.item(), by.item(), bz.item()
            return bx, by, bz

        return field
