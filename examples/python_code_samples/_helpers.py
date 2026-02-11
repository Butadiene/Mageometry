"""
Shared helpers for example scripts.

Provides robust IGRF and T96 wrappers that work with both scalar and
array inputs, falling back to a loop-based vectorizer when the native
vectorized API is unavailable.
"""

import numpy as np
import geopack
from geopack import t96_vectorized


def _loop_vectorize_xyz(func, x, y, z, *args):
    """
    Fallback vectorizer for funcs returning (bx,by,bz) given (x,y,z).
    Supports scalar or array inputs.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # Scalar
    if x.shape == () and y.shape == () and z.shape == ():
        return func(*args, float(x), float(y), float(z))

    # Array (broadcast allowed)
    x, y, z = np.broadcast_arrays(x, y, z)

    bx = np.empty_like(x, dtype=float)
    by = np.empty_like(x, dtype=float)
    bz = np.empty_like(x, dtype=float)

    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        bx[idx], by[idx], bz[idx] = func(*args, float(x[idx]), float(y[idx]), float(z[idx]))
        it.iternext()

    return bx, by, bz


def igrf_internal_gsm(x, y, z):
    """
    Internal field (IGRF) in GSM Cartesian [nT].
    Uses vectorized API if available; otherwise loops.
    """
    if hasattr(geopack, "igrf_gsm_vectorized"):
        return geopack.igrf_gsm_vectorized(x, y, z)
    if hasattr(geopack, "igrf_gsm"):
        return _loop_vectorize_xyz(geopack.igrf_gsm, x, y, z)
    raise AttributeError("geopack has no igrf_gsm / igrf_gsm_vectorized.")


def b_igrf_plus_t96_vectorized(parmod, ps, x, y, z):
    """
    Total magnetic field in GSM Cartesian [nT]:
        B_total = B_IGRF (internal) + B_T96 (external)

    Signature matches Tsyganenko model funcs used by the geometry utilities:
        func(parmod, ps, x, y, z) -> (bx, by, bz)
    """
    bix, biy, biz = igrf_internal_gsm(x, y, z)  # internal
    bex, bey, bez = t96_vectorized(parmod, ps, x, y, z)  # external
    return bix + bex, biy + bey, biz + bez


def default_params():
    """
    Return canonical (ut, ps, parmod) values for examples.

    Returns
    -------
    ut : float
        Unix timestamp (1970-01-01 00:00:00).
    ps : float
        Dipole tilt angle [rad] from geopack.recalc(ut).
    parmod : list
        T96 parameters [Pdyn, Dst, ByIMF, BzIMF, 0, ...].
    """
    ut = 0.0
    ps = geopack.recalc(ut)
    parmod = [2.0, -18.0, 2.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return ut, ps, parmod
