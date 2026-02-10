import os
import sys
import unittest
import datetime
import importlib
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Allowed ULP tolerance (configurable via environment variable)
MAXULP = int(os.environ.get("GEOPACK_MAXULP", "32"))
WARNULP = int(os.environ.get("GEOPACK_WARNULP", "8"))


def ut_seconds(dt: datetime.datetime) -> float:
    epoch = datetime.datetime(1970, 1, 1)
    return (dt - epoch).total_seconds()


def is_scalar_like(a) -> bool:
    if np.isscalar(a):
        return True
    if isinstance(a, np.ndarray) and a.shape == ():
        return True
    return False


def infer_scalar_status(xf, yf, zf, r0: float, rlim: float, steps: int, maxloop: int) -> int:
    r = float(np.sqrt(xf * xf + yf * yf + zf * zf))
    ryz = float(yf * yf + zf * zf)
    if r <= r0 + 1e-6:
        return 0
    if (r >= rlim) or (ryz >= 1600.0) or (xf >= 20.0):
        return 1
    if steps >= maxloop:
        return 2
    return 99


def extract_vectorized_path(masked_arr_2d: np.ma.MaskedArray, i: int) -> np.ndarray:
    valid_idx = np.where(~masked_arr_2d.mask[i])[0]
    return masked_arr_2d.data[i, valid_idx]


def _ulp_map_u64(x: np.ndarray) -> np.ndarray:
    """
    Convert float64 to monotonically increasing uint64 (for ULP distance calculation).
    - negative: bitwise NOT
    - positive: set the sign bit
    """
    x = np.asarray(x, dtype=np.float64)
    u = x.view(np.uint64)
    signbit = np.uint64(0x8000000000000000)
    return np.where((u & signbit) != 0, ~u, u | signbit)


def max_ulp_diff(actual, desired):
    """
    Return the maximum ULP difference between actual/desired and its flat index.
    """
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(desired, dtype=np.float64)
    if a.shape != b.shape:
        raise AssertionError(f"shape mismatch: actual{a.shape} vs desired{b.shape}")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise AssertionError("non-finite values detected")

    oa = _ulp_map_u64(a)
    ob = _ulp_map_u64(b)
    diff = np.where(oa >= ob, oa - ob, ob - oa)  # uint64

    maxdiff = int(diff.max())
    flat_idx = int(diff.reshape(-1).argmax())
    return maxdiff, flat_idx


def assert_close_ulps(actual, desired, *, maxulp: int, name: str, where: str = ""):
    """
    Pass/fail check using eps-scaled (relative + absolute) tolerance.
    Also computes max ULP difference for WARN/FAIL messages.

    Avoids the problem of ULP instability near zero.
    """
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(desired, dtype=np.float64)

    if a.shape != b.shape:
        raise AssertionError(f"{name}{(' '+where) if where else ''}: shape mismatch {a.shape} vs {b.shape}")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise AssertionError(f"{name}{(' '+where) if where else ''}: non-finite values detected")

    # ---- 1) ULP difference (for display) ----
    md, fi = max_ulp_diff(a, b)

    # ---- 2) Eps-scaled pass/fail check (primary criterion) ----
    eps = np.finfo(np.float64).eps  # 2.22e-16
    # Scale: use max(1, |a|, |b|) to avoid being too strict near zero
    scale = np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))
    tol = (maxulp * eps) * scale
    diff = np.abs(a - b)

    if np.any(diff > tol):
        aa = a.reshape(-1)
        bb = b.reshape(-1)
        dd = diff.reshape(-1)
        tt = tol.reshape(-1)
        # Info when eps-scaled check fails (ULP difference included)
        raise AssertionError(
            f"{name}{(' '+where) if where else ''}: eps-scaled mismatch. "
            f"max ulp diff={md} (limit {maxulp}), "
            f"at flat idx={fi}, actual={aa[fi]!r}, desired={bb[fi]!r}, "
            f"absdiff={dd[fi]:.3e}, tol={tt[fi]:.3e}"
        )

    # ---- WARN: display only when ULP difference is large ----
    if md > WARNULP:
        print(f"[WARN] {name}{(' '+where) if where else ''}: max ulp diff={md} (> {WARNULP})")



class TestTraceEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import geopack.geopack as gp
        tv = importlib.import_module("geopack.vectorized.trace")

        cls.gp = gp
        cls.tv = tv

        # Fix global state
        ut = ut_seconds(datetime.datetime(2020, 3, 20, 0, 0, 0))
        gp.recalc(ut)

        # --- Eliminate model differences (force vectorized side to use scalar model calls) ---
        def external_scalar(exname: str, parmod, ps: float, x, y, z, **kwargs):
            if is_scalar_like(x):
                return gp.call_external_model(exname, parmod, ps, float(x), float(y), float(z))
            x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
            bx = np.empty_like(x, dtype=np.float64)
            by = np.empty_like(y, dtype=np.float64)
            bz = np.empty_like(z, dtype=np.float64)
            for i in range(len(x)):
                bx[i], by[i], bz[i] = gp.call_external_model(
                    exname, parmod, ps, float(x[i]), float(y[i]), float(z[i])
                )
            return bx, by, bz

        def internal_scalar(inname: str, x, y, z, **kwargs):
            if is_scalar_like(x):
                return gp.call_internal_model(inname, float(x), float(y), float(z))
            x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
            bx = np.empty_like(x, dtype=np.float64)
            by = np.empty_like(y, dtype=np.float64)
            bz = np.empty_like(z, dtype=np.float64)
            for i in range(len(x)):
                bx[i], by[i], bz[i] = gp.call_internal_model(
                    inname, float(x[i]), float(y[i]), float(z[i])
                )
            return bx, by, bz

        tv.call_external_model_vectorized = external_scalar
        tv.call_internal_model_vectorized = internal_scalar

        if not hasattr(tv, "trace"):
            raise AttributeError("geopack.vectorized.trace does not have a trace function.")

    def run_scalar_one(self, xi, yi, zi, dir, rlim, r0, parmod, exname, inname, maxloop):
        gp = self.gp
        xf, yf, zf, xx, yy, zz = gp.trace(
            xi, yi, zi, dir,
            rlim=rlim, r0=r0,
            parmod=parmod,
            exname=exname, inname=inname,
            maxloop=maxloop
        )
        steps = len(xx) - 1
        status = infer_scalar_status(xf, yf, zf, r0=r0, rlim=rlim, steps=steps, maxloop=maxloop)
        return (xf, yf, zf), (xx, yy, zz), steps, status

    def run_vectorized_batch(self, xi, yi, zi, dir, rlim, r0, parmod, exname, inname, maxloop):
        tv = self.tv
        xf, yf, zf, xx, yy, zz, status = tv.trace(
            xi, yi, zi,
            dir=dir, rlim=rlim, r0=r0,
            parmod=parmod,
            exname=exname, inname=inname,
            maxloop=maxloop,
            return_full_path=True
        )
        return (xf, yf, zf), (xx, yy, zz), status

    def test_equivalence(self):
        rlim = 10.0
        r0 = 1.0
        maxloop = 800
        exname = "t89"
        parmod = 2

        points = np.array([
            [2.0,  0.1,  0.0],
            [3.0,  1.0,  0.5],
            [4.5, -1.2,  0.8],
            [6.0,  0.5, -0.3],
            [8.0, -0.2,  0.1],
            [-2.5, 0.4,  0.2],
        ], dtype=np.float64)

        for inname in ("dipole", "igrf"):
            for dir in (+1.0, -1.0):
                xi = points[:, 0]
                yi = points[:, 1]
                zi = points[:, 2]

                (vxf, vyf, vzf), (vxx_m, vyy_m, vzz_m), vstatus = self.run_vectorized_batch(
                    xi, yi, zi,
                    dir=dir, rlim=rlim, r0=r0,
                    parmod=parmod, exname=exname, inname=inname,
                    maxloop=maxloop
                )

                for i in range(points.shape[0]):
                    where = f"(inname={inname}, dir={dir}, i={i}, p={points[i].tolist()})"
                    with self.subTest(where=where):
                        (sxf, syf, szf), (sxx, syy, szz), ssteps, sstat = self.run_scalar_one(
                            float(xi[i]), float(yi[i]), float(zi[i]),
                            dir=dir, rlim=rlim, r0=r0,
                            parmod=parmod, exname=exname, inname=inname,
                            maxloop=maxloop
                        )

                        vxx = extract_vectorized_path(vxx_m, i)
                        vyy = extract_vectorized_path(vyy_m, i)
                        vzz = extract_vectorized_path(vzz_m, i)
                        vsteps = len(vxx) - 1

                        # Require exact match here (to detect algorithm differences)
                        self.assertEqual(vsteps, ssteps, f"Step mismatch {where}: vec={vsteps} scalar={ssteps}")
                        self.assertEqual(int(vstatus[i]), int(sstat), f"Status mismatch {where}")

                        # Coordinates: allow ULP tolerance
                        assert_close_ulps([vxf[i], vyf[i], vzf[i]], [sxf, syf, szf],
                                          maxulp=MAXULP, name="endpoint", where=where)
                        assert_close_ulps(vxx, sxx, maxulp=MAXULP, name="path_x", where=where)
                        assert_close_ulps(vyy, syy, maxulp=MAXULP, name="path_y", where=where)
                        assert_close_ulps(vzz, szz, maxulp=MAXULP, name="path_z", where=where)


if __name__ == "__main__":
    unittest.main(verbosity=2)
