"""Sample masked fields and Cartesian FAC without rendering dependencies."""

import numpy as np

from ..geometry import field_aligned_current_density
from ..io import GriddedField


def _masked_field(field, mask):
    """Skip excluded regions and undefined frame/stencil coordinates."""
    def evaluate(x, y, z):
        coords = np.broadcast_arrays(x, y, z)
        keep = np.all(np.isfinite(coords), axis=0)
        if mask is not None:
            keep &= ~np.broadcast_to(np.asarray(mask(*coords), dtype=bool), coords[0].shape)
        result = tuple(np.full(coords[0].shape, np.nan) for _ in range(3))
        if np.any(keep):
            components = field(*(c[keep] for c in coords))
            for out, component in zip(result, components):
                out[keep] = component
        return result

    return evaluate


def _preview_indices(shape, max_points):
    """Bound preview size while preserving domain endpoints on every axis."""
    counts = np.asarray(shape, dtype=int)
    if np.any(counts < 3):
        raise ValueError("FAC requires at least three grid nodes on every axis.")
    if max_points is not None:
        if not np.isscalar(max_points) or not np.isfinite(max_points) or max_points < 27:
            raise ValueError("max_points must be an integer >= 27 or None.")
        if int(max_points) != max_points:
            raise ValueError("max_points must be an integer >= 27 or None.")
        if np.prod(counts) > max_points:
            ratio = (max_points / np.prod(counts)) ** (1 / 3)
            counts = np.maximum(3, np.floor((counts - 1) * ratio).astype(int) + 1)
            while np.prod(counts) > max_points:
                axis = int(np.argmax(counts))
                counts[axis] -= 1
    return tuple(np.linspace(0, n - 1, k, dtype=int) for n, k in zip(shape, counts))


def _sample_fac(data, field=None, delta=None, max_points=120000, mask=None):
    """Sample a preview and compute FAC without constructing any VTK objects."""
    indices = _preview_indices(data.shape, max_points)
    axes = tuple(a[i] for a, i in zip((data.x, data.y, data.z), indices))
    b = data.b[np.ix_(*indices)].astype(float, copy=True)
    coords = np.meshgrid(*axes, indexing='ij')
    if mask is not None:
        b[np.broadcast_to(np.asarray(mask(*coords), dtype=bool), b.shape[:-1])] = np.nan

    if field is not None:
        if delta is None:
            delta = tuple(np.min(np.diff(a)) / 2 for a in axes)

        masked_field = _masked_field(field, mask)
        fac = field_aligned_current_density(masked_field, *coords, delta=delta)
        valid = np.all(np.isfinite(b), axis=-1)
        b = np.stack(np.broadcast_arrays(*masked_field(*coords), coords[0])[:3], axis=-1)
        b = np.where(valid[..., None], b, np.nan)
        fac = np.where(valid, fac, np.nan)
    else:
        if delta is not None:
            raise ValueError("delta requires field; grid FAC uses the preview's axis spacing.")
        # Nonuniform Cartesian central differences, with no extrapolated
        # boundary values. Require all six neighbouring samples to be valid.
        dx = np.gradient(b, axes[0], axis=0, edge_order=2)
        dy = np.gradient(b, axes[1], axis=1, edge_order=2)
        dz = np.gradient(b, axes[2], axis=2, edge_order=2)
        curl = np.stack((dy[..., 2] - dz[..., 1], dz[..., 0] - dx[..., 2],
                         dx[..., 1] - dy[..., 0]), axis=-1)
        valid = np.all(np.isfinite(b), axis=-1)
        stencil_valid = valid.copy()
        for axis in range(3):
            stencil_valid &= np.roll(valid, 1, axis) & np.roll(valid, -1, axis)
            boundary = [slice(None)] * 3
            boundary[axis] = [0, -1]
            stencil_valid[tuple(boundary)] = False
        magnitude = np.linalg.norm(b, axis=-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            fac = np.sum(curl * (b / magnitude[..., None]), axis=-1)
        fac = np.where(stencil_valid & (magnitude > 0), fac, np.nan)

    preview = GriddedField(*axes, *np.moveaxis(b, -1, 0))
    return preview, fac
