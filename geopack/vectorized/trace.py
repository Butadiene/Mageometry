"""
Vectorized magnetic field line tracing that matches the original scalar geopack.trace()
as closely as possible (same stopping logic, same ds-handling semantics, same step count
per-trace in principle).

Key "scalar-like" requirements implemented here:
- step()'s adaptive ds changes DO NOT carry over to the next outer loop iteration
  (the original scalar step() does not return ds, so those changes are local only).
- ad sign logic matches scalar (NO extra flip for dir<0).
- ds adjustment in trace() matches scalar condition:
    if (r >= rr) | (r > 5): pass
    else: adjust ds based on r (>=3 => ds=dir, else ds=dir*al)
  i.e., we adjust only when (r < rr) & (r <= 5).
- No outer-boundary-near ds clipping (not in scalar).
- Loop limit matches scalar: while l < maxloop
- On non-convergence in step(): scalar prints and returns None (effectively breaks caller).
  Here we raise RuntimeError to mimic "hard failure" behavior in batch mode.

IMPORTANT:
- To get truly identical results, you should also avoid "vectorized field models"
  that may change floating-point ordering. By default, strict_scalar_models=True,
  which evaluates the underlying models pointwise (loop) for exactness.
- You must call geopack.recalc(ut, ...) before tracing (same as scalar).
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Union
# Field functions (scalar)
from ..geopack import dip, igrf_gsm

# Optional vectorized internal model
try:
    from .igrf import igrf_gsm as igrf_gsm_vectorized  # type: ignore
except Exception:
    igrf_gsm_vectorized = None

from .models import (
    t89 as t89_vectorized,
    t96 as t96_vectorized,
    t01 as t01_vectorized,
    t04 as t04_vectorized,
)

def _as_1d_float64(a) -> np.ndarray:
    return np.atleast_1d(a).astype(np.float64)


def call_external_model(
    exname: str,
    parmod,
    ps: float,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    strict_scalar_models: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    External field model dispatcher.
    If strict_scalar_models=True, always evaluates the scalar model point-by-point,
    even if a vectorized implementation exists, to maximize bitwise-identical behavior.
    """
    ex = exname.lower()

    if ex == "t89":
        if (not strict_scalar_models) and (t89_vectorized is not None):
            return t89_vectorized(parmod, ps, x, y, z)
        from ..models.t89 import t89  # scalar
        bx = np.empty_like(x)
        by = np.empty_like(y)
        bz = np.empty_like(z)
        for i in range(x.size):
            bx[i], by[i], bz[i] = t89(parmod, ps, float(x[i]), float(y[i]), float(z[i]))
        return bx, by, bz

    if ex == "t96":
        if (not strict_scalar_models) and (t96_vectorized is not None):
            return t96_vectorized(parmod, ps, x, y, z)
        from ..models.t96 import t96  # scalar
        bx = np.empty_like(x)
        by = np.empty_like(y)
        bz = np.empty_like(z)
        for i in range(x.size):
            bx[i], by[i], bz[i] = t96(parmod, ps, float(x[i]), float(y[i]), float(z[i]))
        return bx, by, bz

    if ex == "t01":
        if (not strict_scalar_models) and (t01_vectorized is not None):
            return t01_vectorized(parmod, ps, x, y, z)
        from ..models.t01 import t01  # scalar
        bx = np.empty_like(x)
        by = np.empty_like(y)
        bz = np.empty_like(z)
        for i in range(x.size):
            bx[i], by[i], bz[i] = t01(parmod, ps, float(x[i]), float(y[i]), float(z[i]))
        return bx, by, bz

    if ex == "t04":
        if (not strict_scalar_models) and (t04_vectorized is not None):
            return t04_vectorized(parmod, ps, x, y, z)
        from ..models.t04 import t04  # scalar
        bx = np.empty_like(x)
        by = np.empty_like(y)
        bz = np.empty_like(z)
        for i in range(x.size):
            bx[i], by[i], bz[i] = t04(parmod, ps, float(x[i]), float(y[i]), float(z[i]))
        return bx, by, bz

    raise ValueError(f"Unknown external field model: {exname}")


def call_internal_model(
    inname: str,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    strict_scalar_models: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Internal field model dispatcher.
    If strict_scalar_models=True, always evaluates scalar function point-by-point.
    """
    inn = inname.lower()

    if inn in ("igrf", "igrf_gsm"):
        if (not strict_scalar_models) and (igrf_gsm_vectorized is not None):
            return igrf_gsm_vectorized(x, y, z)

        bx = np.empty_like(x)
        by = np.empty_like(y)
        bz = np.empty_like(z)
        for i in range(x.size):
            bx[i], by[i], bz[i] = igrf_gsm(float(x[i]), float(y[i]), float(z[i]))
        return bx, by, bz

    if inn in ("dipole", "dip"):
        bx = np.empty_like(x)
        by = np.empty_like(y)
        bz = np.empty_like(z)
        for i in range(x.size):
            bx[i], by[i], bz[i] = dip(float(x[i]), float(y[i]), float(z[i]))
        return bx, by, bz

    raise ValueError(f"Unknown internal field model: {inname}")


def rhand(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    parmod,
    exname: str,
    inname: str,
    ds3: Union[float, np.ndarray],
    *,
    strict_scalar_models: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized version of scalar rhand():
      bxgsm,bygsm,bzgsm = external(...)
      hxgsm,hygsm,hzgsm = internal(...)
      b = ds3 / sqrt(bx^2 + by^2 + bz^2)
      r = B * b
    """
    from .. import geopack  # to read geopack.psi as scalar does
    ps = geopack.psi

    bxgsm, bygsm, bzgsm = call_external_model(
        exname, parmod, ps, x, y, z, strict_scalar_models=strict_scalar_models
    )
    hxgsm, hygsm, hzgsm = call_internal_model(
        inname, x, y, z, strict_scalar_models=strict_scalar_models
    )

    bx = bxgsm + hxgsm
    by = bygsm + hygsm
    bz = bzgsm + hzgsm

    # Scalar code does NOT guard division by zero. We match that.
    bmag = np.sqrt(bx * bx + by * by + bz * bz)
    b = ds3 / bmag

    r1 = bx * b
    r2 = by * b
    r3 = bz * b
    return r1, r2, r3


def step(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    ds_array: np.ndarray,
    errin: float,
    parmod,
    exname: str,
    inname: str,
    active_mask: np.ndarray,
    *,
    strict_scalar_models: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized step() that matches scalar semantics:
    - ds adaptation happens locally inside this call only
      (ds_array is NOT modified and NOT returned/updated).
    - If any active trace does not converge in max_adapt_iter (=100),
      scalar prints and returns None (caller breaks). Here we raise RuntimeError.
    """
    if errin <= 0:
        raise ValueError("errin must be > 0 (same as scalar).")

    active_idx = np.where(active_mask)[0]
    n_active = active_idx.size
    if n_active == 0:
        return x, y, z

    # Local working copies
    x_act = x[active_mask].copy()
    y_act = y[active_mask].copy()
    z_act = z[active_mask].copy()

    # IMPORTANT: local ds, do not write back to ds_array
    ds_local = ds_array[active_mask].copy()

    converged = np.zeros(n_active, dtype=bool)
    max_adapt_iter = 100  # scalar maxloop in step()

    for _ in range(max_adapt_iter):
        if np.all(converged):
            break

        work = ~converged
        ds = ds_local[work]
        xw = x_act[work]
        yw = y_act[work]
        zw = z_act[work]

        ds3 = -ds / 3.0

        # RK5 stages (exactly as scalar step())
        k1x, k1y, k1z = rhand(
            xw, yw, zw, parmod, exname, inname, ds3, strict_scalar_models=strict_scalar_models
        )

        k2x, k2y, k2z = rhand(
            xw + k1x,
            yw + k1y,
            zw + k1z,
            parmod,
            exname,
            inname,
            ds3,
            strict_scalar_models=strict_scalar_models,
        )

        k3x, k3y, k3z = rhand(
            xw + 0.5 * (k1x + k2x),
            yw + 0.5 * (k1y + k2y),
            zw + 0.5 * (k1z + k2z),
            parmod,
            exname,
            inname,
            ds3,
            strict_scalar_models=strict_scalar_models,
        )

        k4x, k4y, k4z = rhand(
            xw + 0.375 * (k1x + 3.0 * k3x),
            yw + 0.375 * (k1y + 3.0 * k3y),
            zw + 0.375 * (k1z + 3.0 * k3z),
            parmod,
            exname,
            inname,
            ds3,
            strict_scalar_models=strict_scalar_models,
        )

        k5x, k5y, k5z = rhand(
            xw + 1.5 * (k1x - 3.0 * k3x + 4.0 * k4x),
            yw + 1.5 * (k1y - 3.0 * k3y + 4.0 * k4y),
            zw + 1.5 * (k1z - 3.0 * k3z + 4.0 * k4z),
            parmod,
            exname,
            inname,
            ds3,
            strict_scalar_models=strict_scalar_models,
        )

        # errcur = |...| + |...| + |...| (L1), scalar
        errcur = (
            np.abs(k1x - 4.5 * k3x + 4.0 * k4x - 0.5 * k5x)
            + np.abs(k1y - 4.5 * k3y + 4.0 * k4y - 0.5 * k5y)
            + np.abs(k1z - 4.5 * k3z + 4.0 * k4z - 0.5 * k5z)
        )

        ok = errcur < errin

        # Update converged subset positions: x += 0.5*(r11 + 4*r41 + r51) in scalar
        if np.any(ok):
            dx = 0.5 * (k1x[ok] + 4.0 * k4x[ok] + k5x[ok])
            dy = 0.5 * (k1y[ok] + 4.0 * k4y[ok] + k5y[ok])
            dz = 0.5 * (k1z[ok] + 4.0 * k4z[ok] + k5z[ok])

            widx = np.where(work)[0]
            conv_local = widx[ok]

            x_act[conv_local] = xw[ok] + dx
            y_act[conv_local] = yw[ok] + dy
            z_act[conv_local] = zw[ok] + dz

            converged[conv_local] = True

        # For non-converged traces, scalar halves ds and repeats (within this step call only)
        if np.any(~ok):
            widx = np.where(work)[0]
            not_conv_local = widx[~ok]
            ds_local[not_conv_local] *= 0.5

    if not np.all(converged):
        # scalar: print('reached maximum loop ...'); return
        # which breaks caller. We'll emulate hard failure:
        raise RuntimeError("reached maximum loop ...")

    # Write back updated positions for active traces ONLY
    x2 = x.copy()
    y2 = y.copy()
    z2 = z.copy()
    x2[active_mask] = x_act
    y2[active_mask] = y_act
    z2[active_mask] = z_act
    return x2, y2, z2


def adjust_step_sizes(
    r: np.ndarray,
    rr_prev: np.ndarray,
    r0: float,
    dir: float,
    ds_array: np.ndarray,
    active_mask: np.ndarray,
) -> None:
    """
    Exactly matches scalar trace() ds update logic:

        if (r >= rr) | (r > 5):
            pass
        else:
            if r >= 3:
                ds = dir
            else:
                fc = 0.2
                if (r-r0) < 0.05: fc = 0.05
                al = fc*(r-r0+0.2)
                ds = dir*al

    Vectorized: apply only where (r < rr_prev) & (r <= 5) & active_mask.
    Otherwise, keep ds_array as-is.
    """
    mask = active_mask & (r < rr_prev) & (r <= 5.0)
    if not np.any(mask):
        return

    mask_mid = mask & (r >= 3.0)
    if np.any(mask_mid):
        ds_array[mask_mid] = dir

    mask_in = mask & (r < 3.0)
    if np.any(mask_in):
        rin = r[mask_in]
        fc = np.full_like(rin, 0.2)
        fc[(rin - r0) < 0.05] = 0.05
        al = fc * (rin - r0 + 0.2)
        ds_array[mask_in] = dir * al


def trace(
    xi: Union[float, np.ndarray],
    yi: Union[float, np.ndarray],
    zi: Union[float, np.ndarray],
    dir: float = 1.0,
    rlim: float = 10.0,
    r0: float = 1.0,
    parmod=2,
    exname: str = "t89",
    inname: str = "igrf",
    maxloop: int = 1000,
    return_full_path: bool = False,
    *,
    strict_scalar_models: bool = True,
    return_nsteps: bool = False,
) -> Tuple:
    """
    Vectorized version of scalar geopack.trace().

    Parameters
    ----------
    xi, yi, zi : float or array_like
        GSM coordinates in Re. Should be outside r0 to start properly.
    dir : float, optional
        Tracing direction: +1 (antiparallel), -1 (parallel). Default 1.0.
    rlim : float, optional
        Outer boundary radius in Re. Default 10.0.
    r0 : float, optional
        Inner boundary sphere radius in Re. Default 1.0.
    parmod : array_like, optional
        Model parameters. Default 2.
    exname : str, optional
        External field model name. Default 't89'.
    inname : str, optional
        Internal field model name. Default 'igrf'.
    maxloop : int, optional
        Maximum number of integration steps. Default 1000.
    return_full_path : bool, optional
        If True, returns masked arrays (n_traces, maxloop+1). Default False.
    strict_scalar_models : bool, optional
        If True, evaluate B models pointwise to maximize exact match. Default True.
    return_nsteps : bool, optional
        If True, also returns per-trace step counts. Default False.

    Returns
    -------
    xf, yf, zf : float or ndarray
        Final position coordinates in GSM.
    status : int or ndarray
        Status codes: 0 = hit inner boundary, 1 = hit outer boundary, 2 = exceeded maxloop.
    xx, yy, zz : ndarray, optional
        Full path arrays (returned only if return_full_path=True).
    nsteps : int or ndarray, optional
        Per-trace step counts (returned only if return_nsteps=True).
    """
    scalar_input = np.isscalar(xi) and np.isscalar(yi) and np.isscalar(zi)

    xi = _as_1d_float64(xi)
    yi = _as_1d_float64(yi)
    zi = _as_1d_float64(zi)

    if not (xi.size == yi.size == zi.size):
        raise ValueError("xi, yi, zi must have the same length.")

    n = xi.size

    # State arrays
    x = xi.copy()
    y = yi.copy()
    z = zi.copy()

    status = np.zeros(n, dtype=np.int32)  # 0 running/inner-hit, 1 outer, 2 maxloop
    active = np.ones(n, dtype=bool)

    # ds initial = 0.5*dir (scalar)
    ds_array = np.full(n, 0.5 * dir, dtype=np.float64)

    # Initial direction check for ad (scalar)
    ds3_init = -(0.5 * dir) / 3.0
    r1, r2, r3 = rhand(
        xi, yi, zi, parmod, exname, inname, ds3_init,
        strict_scalar_models=strict_scalar_models
    )
    br_like = xi * r1 + yi * r2 + zi * r3  # matches scalar test quantity
    ad = np.where(br_like < 0.0, -0.01, 0.01)  # NO extra dir flip (scalar)

    rr = np.sqrt(xi * xi + yi * yi + zi * zi) + ad  # scalar rr initialization

    # Optional full path storage
    if return_full_path:
        # scalar returns 1D arrays per trace, variable length.
        # Here we store a masked (n, maxloop+1) like your approach.
        xx = np.ma.masked_all((n, maxloop + 1), dtype=np.float64)
        yy = np.ma.masked_all((n, maxloop + 1), dtype=np.float64)
        zz = np.ma.masked_all((n, maxloop + 1), dtype=np.float64)
        xx[:, 0] = x
        yy[:, 0] = y
        zz[:, 0] = z

    # Per-trace accepted step count (how many step() calls succeeded)
    nsteps = np.zeros(n, dtype=np.int32)

    # Scalar uses err=0.001 constant (named err)
    errin = 0.001

    # Scalar loop: while l < maxloop:
    # Here, we perform up to maxloop accepted steps per trace.
    # In each batch iteration, active traces attempt exactly one step() call.
    for l in range(maxloop):
        if not np.any(active):
            break

        r2 = x * x + y * y + z * z
        ryz = y * y + z * z
        r = np.sqrt(r2)

        # Save current position (scalar sets xr,yr,zr but (as written) it's the current point)
        xr = x.copy()
        yr = y.copy()
        zr = z.copy()

        # Outer boundary check BEFORE step (scalar)
        mask_outer = active & ((r >= rlim) | (ryz >= 1600.0) | (x >= 20.0))
        if np.any(mask_outer):
            status[mask_outer] = 1
            active[mask_outer] = False

        if not np.any(active):
            break

        # Inner boundary crossing check (scalar)
        # NOTE: scalar uses rr (previous radial) and current r, but uses current xr,yr,zr
        # which effectively makes interpolation a no-op; we preserve that behavior.
        mask_inner = active & (r < r0) & (rr > r)
        if np.any(mask_inner):
            t = (r0 - r[mask_inner]) / (rr[mask_inner] - r[mask_inner])
            x[mask_inner] = x[mask_inner] - (x[mask_inner] - xr[mask_inner]) * t
            y[mask_inner] = y[mask_inner] - (y[mask_inner] - yr[mask_inner]) * t
            z[mask_inner] = z[mask_inner] - (z[mask_inner] - zr[mask_inner]) * t
            status[mask_inner] = 0
            active[mask_inner] = False

        if not np.any(active):
            break

        rr_prev = rr.copy()

        # ds update based on scalar logic (uses rr_prev)
        adjust_step_sizes(r, rr_prev, r0, dir, ds_array, active)

        # scalar sets rr=r (for next iteration)
        rr[active] = r[active]

        # Do one step() call for active traces
        x_before = x.copy()
        y_before = y.copy()
        z_before = z.copy()

        x, y, z = step(
            x, y, z,
            ds_array, errin,
            parmod, exname, inname,
            active,
            strict_scalar_models=strict_scalar_models,
        )

        # Ensure inactive traces didn't move (defensive; step() only updates active anyway)
        x[~active] = x_before[~active]
        y[~active] = y_before[~active]
        z[~active] = z_before[~active]

        nsteps[active] += 1

        if return_full_path:
            # Store positions at column l+1 (since 0 is initial)
            xx[active, l + 1] = x[active]
            yy[active, l + 1] = y[active]
            zz[active, l + 1] = z[active]

    # Any traces still active after maxloop => status=2 (scalar would just exit with l==maxloop)
    still = active & (status == 0)
    if np.any(still):
        status[still] = 2
        active[still] = False

    xf = x
    yf = y
    zf = z

    # Scalar input => scalar output
    if scalar_input:
        xf = float(xf[0])
        yf = float(yf[0])
        zf = float(zf[0])
        st = int(status[0])
        if return_full_path:
            # Return 1D arrays for scalar case (trim masked)
            valid = ~xx.mask[0]
            xx1 = xx.data[0][valid]
            yy1 = yy.data[0][valid]
            zz1 = zz.data[0][valid]
            if return_nsteps:
                return xf, yf, zf, xx1, yy1, zz1, st, int(nsteps[0])
            return xf, yf, zf, xx1, yy1, zz1, st
        else:
            if return_nsteps:
                return xf, yf, zf, st, int(nsteps[0])
            return xf, yf, zf, st

    # Vector input
    if return_full_path:
        if return_nsteps:
            return xf, yf, zf, xx, yy, zz, status, nsteps
        return xf, yf, zf, xx, yy, zz, status

    if return_nsteps:
        return xf, yf, zf, status, nsteps
    return xf, yf, zf, status