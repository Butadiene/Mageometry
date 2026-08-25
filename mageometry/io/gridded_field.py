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


def region_slices(axes, region=None, stride=1):
    """
    Index slices selecting a bounding box (and stride) on rectilinear axes.

    Parameters
    ----------
    axes : sequence of 1D arrays
        The (x, y, z) coordinate axes.
    region : sequence, optional
        ``((xmin, xmax), (ymin, ymax), (zmin, zmax))``; an entry of ``None``
        keeps the full axis. Bounds are inclusive.
    stride : int or tuple of int, optional
        Keep every ``stride``-th node per axis.

    Returns
    -------
    tuple of slice
        One slice per axis, usable on the axes and on ``(nx, ny, nz)`` data.
    """
    if np.isscalar(stride):
        stride = (stride,) * len(axes)
    if region is None:
        region = (None,) * len(axes)
    if len(region) != len(axes) or len(stride) != len(axes):
        raise ValueError("region and stride must have one entry per axis.")
    slices = []
    for ax, bounds, st in zip(axes, region, stride):
        if bounds is None:
            lo, hi = 0, ax.size
        else:
            bmin, bmax = bounds
            lo = int(np.searchsorted(ax, bmin, side='left'))
            hi = int(np.searchsorted(ax, bmax, side='right'))
        n_sel = len(range(lo, hi, int(st)))
        if n_sel < 2:
            raise ValueError(
                f"Region {bounds} with stride {st} selects {n_sel} node(s) on an axis "
                f"spanning [{ax[0]:g}, {ax[-1]:g}]; at least 2 are needed."
            )
        slices.append(slice(lo, hi, int(st)))
    return tuple(slices)


class FieldSeries:
    """
    A lazily loaded time series of `GriddedField` objects.

    Steps are read only when accessed, so a long series of large grids never
    has to fit in memory at once. Build one from your own per-step files with
    `FieldSeries.from_files`; `load_xdmf_series` returns one for XDMF data.

    Parameters
    ----------
    steps : sequence of (time, source, loader)
        ``time`` (float or None), a description of the step's origin, and a
        zero-argument callable returning the `GriddedField`.

    Attributes
    ----------
    times : ndarray
        Time value of each step (NaN where unknown).
    sources : list
        Origin of each step (e.g. file path).
    """

    def __init__(self, steps):
        self._steps = [tuple(s) for s in steps]
        self.times = np.array([t if t is not None else np.nan
                               for t, _, _ in self._steps], dtype=np.float64)
        self.sources = [s for _, s, _ in self._steps]

    @classmethod
    def from_files(cls, paths, loader, times=None, **loader_kwargs):
        """
        Series from one file per step and a reader function.

        Parameters
        ----------
        paths : sequence of str
            One file per step, in time order.
        loader : callable
            ``loader(path, **loader_kwargs) -> GriddedField`` — a bundled
            reader (`load_xdmf`, `load_hdf5`) or your own ``load_<format>``.
        times : sequence of float, optional
            Time value per step. Default: unknown (NaN).
        **loader_kwargs
            Passed to ``loader`` for every step.
        """
        paths = list(paths)
        if times is None:
            times = [None] * len(paths)
        elif len(times) != len(paths):
            raise ValueError("times must have one entry per path.")

        def make(p):
            return lambda: loader(p, **loader_kwargs)

        return cls([(t, p, make(p)) for t, p in zip(times, paths)])

    def __len__(self):
        return len(self._steps)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return FieldSeries(self._steps[i])
        n = len(self._steps)
        if not -n <= i < n:
            raise IndexError(f"step {i} out of range for series of {n} steps")
        return self._steps[i][2]()

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def index_at(self, time):
        """Index of the step whose time is closest to ``time``."""
        if np.all(np.isnan(self.times)):
            raise ValueError("This series carries no time values; index by step.")
        return int(np.nanargmin(np.abs(self.times - time)))

    def at(self, time):
        """Load the step whose time is closest to ``time``."""
        return self[self.index_at(time)]

    def __repr__(self):
        if len(self) and not np.all(np.isnan(self.times)):
            span = f", t=[{np.nanmin(self.times):g}, {np.nanmax(self.times):g}]"
        else:
            span = ""
        return f"{type(self).__name__}(n_steps={len(self)}{span})"


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

    def subvolume(self, region=None, stride=1):
        """
        Extract a sub-box (optionally coarsened) as a new `GriddedField`.

        Parameters
        ----------
        region : sequence, optional
            ``((xmin, xmax), (ymin, ymax), (zmin, zmax))`` in grid
            coordinates, bounds inclusive; ``None`` per axis keeps the full
            extent.
        stride : int or tuple of int, optional
            Keep every ``stride``-th node per axis. Default 1.

        Returns
        -------
        GriddedField
            A copy of the selected nodes (the original is left untouched).
        """
        sx, sy, sz = region_slices((self.x, self.y, self.z), region, stride)
        meta = dict(self.metadata, subvolume=(region, stride))
        return GriddedField(self.x[sx], self.y[sy], self.z[sz],
                            self.bx[sx, sy, sz], self.by[sx, sy, sz],
                            self.bz[sx, sy, sz], metadata=meta)

    def divergence(self, relative=True):
        """
        Finite-difference divergence of the field on the grid, as a sanity check.

        A physical MHD field is (nearly) divergence-free, so a large result
        almost always means the arrays were assembled wrongly: axes in the
        wrong order, components permuted, or a wrong grid spacing.

        Parameters
        ----------
        relative : bool, optional
            If True (default) return the dimensionless
            ``|div B| * h / |B|`` with ``h`` the mean grid spacing. Its
            median away from the planet and the grid edges is ~1e-3-1e-2 for
            correctly assembled data and ~0.1 or more when components are
            permuted or sign-flipped or the axes are transposed. If False
            return ``div B`` in field-unit / length-unit.

        Returns
        -------
        ndarray of shape ``(nx, ny, nz)``
        """
        b = self.b.astype(np.float64, copy=False)
        div = (np.gradient(b[..., 0], self.x, axis=0)
               + np.gradient(b[..., 1], self.y, axis=1)
               + np.gradient(b[..., 2], self.z, axis=2))
        if not relative:
            return div
        h = np.mean([np.mean(np.diff(self.x)), np.mean(np.diff(self.y)),
                     np.mean(np.diff(self.z))])
        bmag = np.sqrt(np.sum(b * b, axis=-1))
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(bmag > 0, np.abs(div) * h / bmag, np.nan)

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
