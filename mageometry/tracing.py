# mageometry/tracing.py
"""
Field line tracing for any ``field(x, y, z) -> (bx, by, bz)`` callable.

`trace_field_lines` integrates the unit-tangent ODE dr/ds = B/|B| with the
same fifth-order Runge-Kutta scheme (and per-step halving error control) used
by geopack, but it is decoupled from the geopack models: the magnetic field is
any callable following the Mageometry field convention, so the same code
traces Tsyganenko fields (`mageometry.geopack_field`) and interpolated
simulation data (`GriddedField.field()`).

Units are the field's own: seeds, step size, radii, and the returned paths
share the length unit of the field's positions. Nothing here assumes Re or
GSM coordinates.

The bitwise-faithful reproduction of the scalar `geopack.trace` (including
its magnetosphere-specific stopping rules) remains available as
`mageometry.geopack.trace_vectorized` for validation of the geopack engine.
"""

import numpy as np

__all__ = [
    "trace_field_lines",
    "FieldLineTrace",
    "STATUS_INNER",
    "STATUS_OUTER",
    "STATUS_MAX_STEPS",
    "STATUS_INVALID",
    "STATUS_STOPPED",
]

STATUS_INNER = 0
"""Reached the inner sphere ``r0``."""
STATUS_OUTER = 1
"""Reached the outer sphere ``rlim`` or the bounding box ``bounds``."""
STATUS_MAX_STEPS = 2
"""Stopped after ``max_steps`` integration steps."""
STATUS_INVALID = 3
"""The field is undefined ahead (non-finite or zero): typically the edge of
the data domain of an interpolated field, or a magnetic null."""
STATUS_STOPPED = 4
"""The user-supplied ``stop`` condition became true."""

_STATUS_NAMES = {
    STATUS_INNER: "inner",
    STATUS_OUTER: "outer",
    STATUS_MAX_STEPS: "max_steps",
    STATUS_INVALID: "invalid",
    STATUS_STOPPED: "stopped",
}


class FieldLineTrace:
    """
    Result of `trace_field_lines`: a batch of traced field lines.

    Paths are stored as 2D arrays of shape ``(n_lines, n_points_max)``,
    padded with NaN beyond each line's last point. Use `path` to get one
    line's trimmed coordinates.

    Attributes
    ----------
    x, y, z : ndarray, shape (n_lines, n_points_max)
        Path coordinates, NaN-padded.
    s : ndarray, shape (n_lines, n_points_max)
        Arc length along each path, measured from the seed point. Zero at
        the seed and increasing along the traced direction. For
        ``direction='both'`` it is signed: negative on the -B side, positive
        on the +B side, so it increases monotonically along the stored path.
    nsteps : ndarray of int, shape (n_lines,)
        Number of integration steps per line (``n_points - 1``).
    start_index : ndarray of int, shape (n_lines,)
        Column index of the seed point in each path. Zero unless
        ``direction='both'``.
    status : ndarray of int, shape (n_lines,)
        Termination code at the end of the traced direction (the +B end for
        ``direction='both'``); see the ``STATUS_*`` constants.
    status_backward : ndarray of int or None
        Termination code at the -B end for ``direction='both'``; None
        otherwise.
    """

    def __init__(self, x, y, z, s, nsteps, start_index, status,
                 status_backward=None):
        self.x = x
        self.y = y
        self.z = z
        self.s = s
        self.nsteps = nsteps
        self.start_index = start_index
        self.status = status
        self.status_backward = status_backward

    @property
    def n_lines(self):
        return self.x.shape[0]

    def path(self, i):
        """Coordinates ``(x, y, z)`` of line ``i`` as trimmed 1D arrays."""
        n = self.nsteps[i] + 1
        return self.x[i, :n], self.y[i, :n], self.z[i, :n]

    def arc_length(self, i):
        """Arc length ``s`` of line ``i`` as a trimmed 1D array."""
        return self.s[i, :self.nsteps[i] + 1]

    @property
    def end(self):
        """Final points ``(x, y, z)`` of all lines, shape (n_lines,) each."""
        idx = np.arange(self.n_lines)
        last = self.nsteps
        return self.x[idx, last], self.y[idx, last], self.z[idx, last]

    @property
    def start(self):
        """Starting points (-B end for ``direction='both'``, else seeds)."""
        idx = np.arange(self.n_lines)
        return self.x[idx, 0], self.y[idx, 0], self.z[idx, 0]

    def __repr__(self):
        counts = {name: int(np.sum(self.status == code))
                  for code, name in _STATUS_NAMES.items()}
        counts = {k: v for k, v in counts.items() if v}
        return (f"FieldLineTrace(n_lines={self.n_lines}, "
                f"max_points={self.x.shape[1]}, status={counts})")


def trace_field_lines(field, x, y, z, direction=1, ds=0.1, *, err=1e-3,
                      r0=None, rlim=None, bounds=None, stop=None,
                      max_steps=1000):
    """
    Trace magnetic field lines through the given seed points.

    Parameters
    ----------
    field : callable
        ``field(x, y, z) -> (bx, by, bz)`` accepting 1D NumPy arrays. For
        interpolated fields use ``fill_value=np.nan`` (the default of
        `GriddedField.field`) so that leaving the data domain terminates the
        line with `STATUS_INVALID` instead of raising.
    x, y, z : float or array_like
        Seed point coordinates (same length unit as the field).
    direction : {1, -1, 'both'}, optional
        ``1`` traces along B, ``-1`` against B, ``'both'`` traces in both
        directions and joins the two halves into one path per seed with the
        seed at ``start_index``. Default 1.
    ds : float, optional
        Nominal step length (arc length per accepted step) in the field's
        length unit. Each step starts from ``ds`` and is halved until the
        embedded error estimate is below ``err * ds``. Default 0.1.
    err : float, optional
        Relative error tolerance per step (dimensionless, relative to
        ``ds``). Default 1e-3.
    r0 : float, optional
        Inner sphere radius: a line entering ``r < r0`` is terminated with
        `STATUS_INNER`, its last point linearly interpolated onto the sphere.
        None (default) disables the check.
    rlim : float, optional
        Outer sphere radius; a line reaching ``r >= rlim`` is terminated with
        `STATUS_OUTER`, its last point interpolated onto the sphere. None
        (default) disables the check.
    bounds : sequence, optional
        ``((xmin, xmax), (ymin, ymax), (zmin, zmax))`` bounding box (e.g.
        ``GriddedField.bounds``). Leaving it terminates the line with
        `STATUS_OUTER`, its last point placed onto the box face. Without it,
        leaving the domain of an interpolated field (NaN fill) is reported as
        `STATUS_INVALID`.
    stop : callable, optional
        ``stop(x, y, z) -> bool array``; lines where it is true are terminated
        at that point with `STATUS_STOPPED`. Evaluated on 1D arrays.
    max_steps : int, optional
        Maximum number of accepted steps per line and direction. Default 1000.

    Returns
    -------
    FieldLineTrace
        Paths, arc lengths, step counts, and termination codes.

    Notes
    -----
    Integration scheme: the fifth-order Runge-Kutta formula of geopack's
    ``step()`` with its L1 error estimate. Unlike `geopack.trace_vectorized`,
    step halving does not depend on any Earth-specific radius, boundary
    crossings are interpolated, and the field is a generic callable.
    """
    x0 = np.atleast_1d(np.asarray(x, dtype=np.float64)).ravel()
    y0 = np.atleast_1d(np.asarray(y, dtype=np.float64)).ravel()
    z0 = np.atleast_1d(np.asarray(z, dtype=np.float64)).ravel()
    if not (x0.size == y0.size == z0.size):
        raise ValueError("x, y, z seeds must have the same length.")

    if ds <= 0:
        raise ValueError("ds must be > 0; use `direction` to choose the sense.")
    if err <= 0:
        raise ValueError("err must be > 0.")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1.")
    if bounds is not None:
        bounds = np.asarray(bounds, dtype=np.float64)
        if bounds.shape != (3, 2) or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError(
                "bounds must be ((xmin, xmax), (ymin, ymax), (zmin, zmax)) "
                "with min < max."
            )

    opts = dict(ds=float(ds), err=float(err), r0=r0, rlim=rlim,
                bounds=bounds, stop=stop, max_steps=int(max_steps))

    if direction == 'both':
        back = _trace_one_direction(field, x0, y0, z0, -1.0, **opts)
        fwd = _trace_one_direction(field, x0, y0, z0, +1.0, **opts)
        return _join_both(back, fwd)
    if direction in (1, -1, 1.0, -1.0):
        res = _trace_one_direction(field, x0, y0, z0, float(direction), **opts)
        x, y, z, s, nsteps, status = res
        n = x0.size
        return FieldLineTrace(x, y, z, s, nsteps, np.zeros(n, dtype=np.int64),
                              status)
    raise ValueError("direction must be 1, -1, or 'both'.")


# ---------------------------------------------------------------------------
# Integration core
# ---------------------------------------------------------------------------

_MAX_HALVINGS = 20


def _rhand(field, x, y, z, ds3):
    """Unit tangent times ds3: ds3 * B/|B| (NaN where B is undefined/zero)."""
    bx, by, bz = field(x, y, z)
    bx = np.asarray(bx, dtype=np.float64)
    by = np.asarray(by, dtype=np.float64)
    bz = np.asarray(bz, dtype=np.float64)
    bmag = np.sqrt(bx * bx + by * by + bz * bz)
    with np.errstate(divide='ignore', invalid='ignore'):
        f = np.where(bmag > 0, ds3 / bmag, np.nan)
    return bx * f, by * f, bz * f


def _rk5_step(field, x, y, z, ds_signed, tol):
    """
    One adaptive RK5 step for a batch of points.

    Each point starts with step ``ds_signed`` and halves it until the error
    estimate is below ``tol`` (at most ``_MAX_HALVINGS`` times). Returns the
    new positions, the accepted (unsigned) step lengths, and an ``ok`` mask;
    points that never converged (non-finite field ahead) keep their input
    position with ``ok=False``.
    """
    n = x.size
    xn, yn, zn = x.copy(), y.copy(), z.copy()
    ds = np.full(n, ds_signed, dtype=np.float64)
    ds_used = np.zeros(n, dtype=np.float64)
    ok = np.zeros(n, dtype=bool)

    for _ in range(_MAX_HALVINGS + 1):
        work = np.where(~ok)[0]
        if work.size == 0:
            break
        xw, yw, zw = x[work], y[work], z[work]
        ds3 = ds[work] / 3.0

        k1x, k1y, k1z = _rhand(field, xw, yw, zw, ds3)
        k2x, k2y, k2z = _rhand(field, xw + k1x, yw + k1y, zw + k1z, ds3)
        k3x, k3y, k3z = _rhand(field,
                               xw + 0.5 * (k1x + k2x),
                               yw + 0.5 * (k1y + k2y),
                               zw + 0.5 * (k1z + k2z), ds3)
        k4x, k4y, k4z = _rhand(field,
                               xw + 0.375 * (k1x + 3.0 * k3x),
                               yw + 0.375 * (k1y + 3.0 * k3y),
                               zw + 0.375 * (k1z + 3.0 * k3z), ds3)
        k5x, k5y, k5z = _rhand(field,
                               xw + 1.5 * (k1x - 3.0 * k3x + 4.0 * k4x),
                               yw + 1.5 * (k1y - 3.0 * k3y + 4.0 * k4y),
                               zw + 1.5 * (k1z - 3.0 * k3z + 4.0 * k4z), ds3)

        errcur = (np.abs(k1x - 4.5 * k3x + 4.0 * k4x - 0.5 * k5x)
                  + np.abs(k1y - 4.5 * k3y + 4.0 * k4y - 0.5 * k5y)
                  + np.abs(k1z - 4.5 * k3z + 4.0 * k4z - 0.5 * k5z))
        # Non-finite errcur (field undefined at a stage point) compares False.
        with np.errstate(invalid='ignore'):
            good = errcur < tol

        if np.any(good):
            g = work[good]
            xn[g] = xw[good] + 0.5 * (k1x[good] + 4.0 * k4x[good] + k5x[good])
            yn[g] = yw[good] + 0.5 * (k1y[good] + 4.0 * k4y[good] + k5y[good])
            zn[g] = zw[good] + 0.5 * (k1z[good] + 4.0 * k4z[good] + k5z[good])
            ds_used[g] = np.abs(ds[g])
            ok[g] = True
        bad = work[~good]
        ds[bad] *= 0.5

    return xn, yn, zn, ds_used, ok


def _terminate(xp, yp, zp, xc, yc, zc, r0, rlim, bounds, stop):
    """
    Classify termination at the current points ``c`` given previous ``p``.

    Returns ``(status, xe, ye, ze, t)``: status is -1 where the line goes on;
    otherwise the end point ``e`` lies on the crossed boundary (interpolated
    between p and c) and ``t`` in [0, 1] is its fraction along p->c.
    """
    n = xc.size
    status = np.full(n, -1, dtype=np.int64)
    t = np.ones(n, dtype=np.float64)

    # Custom stop: terminates exactly at the current point.
    if stop is not None:
        st = np.asarray(stop(xc, yc, zc), dtype=bool)
        if st.shape != (n,):
            raise ValueError("stop(x, y, z) must return a bool array of the seed length.")
        status[st] = STATUS_STOPPED

    dx, dy, dz = xc - xp, yc - yp, zc - zp
    r_c = np.sqrt(xc * xc + yc * yc + zc * zc)

    if r0 is not None:
        hit = (status < 0) & (r_c < r0)
        if np.any(hit):
            t[hit] = _sphere_fraction(xp[hit], yp[hit], zp[hit],
                                      dx[hit], dy[hit], dz[hit], r0, entering=True)
            status[hit] = STATUS_INNER

    if rlim is not None:
        hit = (status < 0) & (r_c >= rlim)
        if np.any(hit):
            t[hit] = _sphere_fraction(xp[hit], yp[hit], zp[hit],
                                      dx[hit], dy[hit], dz[hit], rlim, entering=False)
            status[hit] = STATUS_OUTER

    if bounds is not None:
        lo, hi = bounds[:, 0], bounds[:, 1]
        t_box = np.ones(n, dtype=np.float64)
        outside = np.zeros(n, dtype=bool)
        for c, p, d, lo_i, hi_i in ((xc, xp, dx, lo[0], hi[0]),
                                    (yc, yp, dy, lo[1], hi[1]),
                                    (zc, zp, dz, lo[2], hi[2])):
            with np.errstate(divide='ignore', invalid='ignore'):
                below = c < lo_i
                above = c > hi_i
                t_lo = np.where(below, (lo_i - p) / d, 1.0)
                t_hi = np.where(above, (hi_i - p) / d, 1.0)
            outside |= below | above
            t_box = np.minimum(t_box, np.where(below, t_lo, t_hi))
        hit = (status < 0) & outside
        if np.any(hit):
            t[hit] = np.clip(np.nan_to_num(t_box[hit], nan=0.0), 0.0, 1.0)
            status[hit] = STATUS_OUTER

    xe = xp + t * dx
    ye = yp + t * dy
    ze = zp + t * dz
    return status, xe, ye, ze, t


def _snap_to_bounds(idx, xc, yc, zc, X, Y, Z, nsteps, status, bounds, ds):
    """
    Reclassify failed steps that ended on the bounding box as `STATUS_OUTER`.

    An interpolated field returns NaN outside its domain, so a line reaches
    the domain edge by repeatedly halving its step (the RK stage points probe
    ahead) until the step fails: the last accepted point then lies within
    ~``ds * 2**-20`` of the edge. When that edge coincides with ``bounds``,
    report it as a boundary hit and snap the point exactly onto the face.
    """
    lo, hi = bounds[:, 0], bounds[:, 1]
    tol = 1e-4 * ds
    on_face = np.zeros(idx.size, dtype=bool)
    for k, c in enumerate((xc, yc, zc)):
        v = c[idx]
        near_lo = np.abs(v - lo[k]) <= tol
        near_hi = np.abs(v - hi[k]) <= tol
        v = np.where(near_lo, lo[k], np.where(near_hi, hi[k], v))
        c[idx] = v
        on_face |= near_lo | near_hi
    hit = idx[on_face]
    if hit.size:
        status[hit] = STATUS_OUTER
        last = nsteps[hit]
        X[hit, last], Y[hit, last], Z[hit, last] = xc[hit], yc[hit], zc[hit]


def _sphere_fraction(xp, yp, zp, dx, dy, dz, radius, entering):
    """
    Fraction t in [0, 1] along the chord p + t*d where it crosses the sphere
    of the given radius: the exact root of |p + t d|^2 = radius^2.

    ``entering=True`` selects the crossing from outside to inside (smaller
    root), ``False`` the crossing from inside to outside (larger root).
    """
    a = dx * dx + dy * dy + dz * dz
    b = 2.0 * (xp * dx + yp * dy + zp * dz)
    c = xp * xp + yp * yp + zp * zp - radius * radius
    disc = np.maximum(b * b - 4.0 * a * c, 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        sq = np.sqrt(disc)
        t = (-b - sq) / (2.0 * a) if entering else (-b + sq) / (2.0 * a)
    return np.clip(np.nan_to_num(t, nan=0.0), 0.0, 1.0)


def _trace_one_direction(field, x0, y0, z0, sign, *, ds, err, r0, rlim,
                         bounds, stop, max_steps):
    n = x0.size
    ncol = max_steps + 1
    X = np.full((n, ncol), np.nan)
    Y = np.full((n, ncol), np.nan)
    Z = np.full((n, ncol), np.nan)
    S = np.full((n, ncol), np.nan)
    X[:, 0], Y[:, 0], Z[:, 0], S[:, 0] = x0, y0, z0, 0.0

    nsteps = np.zeros(n, dtype=np.int64)
    status = np.full(n, -1, dtype=np.int64)
    active = np.ones(n, dtype=bool)

    xc, yc, zc = x0.copy(), y0.copy(), z0.copy()
    xp, yp, zp = x0.copy(), y0.copy(), z0.copy()
    tol = err * ds

    def classify(idx):
        """Apply termination checks to active lines ``idx`` at their current point."""
        st, xe, ye, ze, t = _terminate(xp[idx], yp[idx], zp[idx],
                                       xc[idx], yc[idx], zc[idx],
                                       r0, rlim, bounds, stop)
        done = st >= 0
        if not np.any(done):
            return
        d = idx[done]
        status[d] = st[done]
        active[d] = False
        # Move the stored last point onto the boundary and fix its arc length.
        xc[d], yc[d], zc[d] = xe[done], ye[done], ze[done]
        last = nsteps[d]
        X[d, last], Y[d, last], Z[d, last] = xe[done], ye[done], ze[done]
        prev_s = np.where(last > 0, S[d, np.maximum(last - 1, 0)], 0.0)
        S[d, last] = prev_s + t[done] * (S[d, last] - prev_s)

    for _ in range(max_steps):
        idx = np.where(active)[0]
        if idx.size == 0:
            break
        classify(idx)
        idx = np.where(active)[0]
        if idx.size == 0:
            break

        xn, yn, zn, ds_used, ok = _rk5_step(field, xc[idx], yc[idx], zc[idx],
                                            sign * ds, tol)
        failed = idx[~ok]
        if failed.size:
            status[failed] = STATUS_INVALID
            active[failed] = False
            if bounds is not None:
                _snap_to_bounds(failed, xc, yc, zc, X, Y, Z, nsteps, status,
                                bounds, ds)
        moved = idx[ok]
        if moved.size:
            xp[moved], yp[moved], zp[moved] = xc[moved], yc[moved], zc[moved]
            xc[moved], yc[moved], zc[moved] = xn[ok], yn[ok], zn[ok]
            nsteps[moved] += 1
            col = nsteps[moved]
            X[moved, col], Y[moved, col], Z[moved, col] = xn[ok], yn[ok], zn[ok]
            S[moved, col] = S[moved, col - 1] + ds_used[ok]

    idx = np.where(active)[0]
    if idx.size:
        classify(idx)
    still = np.where(active)[0]
    status[still] = STATUS_MAX_STEPS

    # Trim unused columns.
    ncol_used = int(nsteps.max()) + 1
    return (X[:, :ncol_used], Y[:, :ncol_used], Z[:, :ncol_used],
            S[:, :ncol_used], nsteps, status)


def _join_both(back, fwd):
    """Join backward (-B) and forward (+B) halves into single paths."""
    xb, yb, zb, sb, nb, stb = back
    xf, yf, zf, sf, nf, stf = fwd
    n = xb.shape[0]
    ntot = nb + nf
    ncol = int(ntot.max()) + 1
    X = np.full((n, ncol), np.nan)
    Y = np.full((n, ncol), np.nan)
    Z = np.full((n, ncol), np.nan)
    S = np.full((n, ncol), np.nan)
    for i in range(n):
        kb, kf = nb[i], nf[i]
        # Backward half reversed (excluding the seed), then the forward half.
        X[i, :kb] = xb[i, kb:0:-1]
        Y[i, :kb] = yb[i, kb:0:-1]
        Z[i, :kb] = zb[i, kb:0:-1]
        S[i, :kb] = -sb[i, kb:0:-1]
        X[i, kb:kb + kf + 1] = xf[i, :kf + 1]
        Y[i, kb:kb + kf + 1] = yf[i, :kf + 1]
        Z[i, kb:kb + kf + 1] = zf[i, :kf + 1]
        S[i, kb:kb + kf + 1] = sf[i, :kf + 1]
    return FieldLineTrace(X, Y, Z, S, ntot, nb.copy(), stf, status_backward=stb)
