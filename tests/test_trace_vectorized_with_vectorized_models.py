# tests/test_trace_vectorized_with_vectorized_models.py
import os
import sys
import unittest
import datetime
import importlib
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Trace integration test tolerance (in Re units)
TRACE_ATOL_RE = float(os.environ.get("GEOPACK_TRACE_ATOL_RE", "1e-5"))  # 1e-5 Re ~ 63m
TRACE_RTOL = float(os.environ.get("GEOPACK_TRACE_RTOL", "0.0"))


def ut_seconds(dt: datetime.datetime) -> float:
    epoch = datetime.datetime(1970, 1, 1)
    return (dt - epoch).total_seconds()


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


def parmod_for(exname: str):
    """
    Return parmod for the given external model.
    t89 uses iopt (int).
    t96/t01/t04 use a 10-element array (matching this project's model implementation).
    """
    ex = exname.lower()
    if ex == "t89":
        return 2
    # e.g. [Pdyn, Dst, ByIMF, BzIMF, W1..W6]
    return np.array([2.0, -20.0, 0.0, -5.0, 0, 0, 0, 0, 0, 0], dtype=np.float64)


class TestTraceVectorizedWithVectorizedModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mageometry.geopack.geopack as gp
        cls.gp = gp

        cls.tv = importlib.import_module("mageometry.geopack.vectorized.trace")

        ut = ut_seconds(datetime.datetime(2020, 3, 20, 0, 0, 0))
        gp.recalc(ut)

        # Ensure vectorized external models are available
        try:
            from mageometry.geopack.vectorized import models as vmodels
        except Exception as e:
            cls.skip_reason = f"mageometry.mageometry.geopack.vectorized.models not available: {e}"
            cls.vmodels = None
        else:
            cls.skip_reason = None
            cls.vmodels = vmodels

        if not hasattr(cls.tv, "trace"):
            raise AttributeError("mageometry.geopack.vectorized.trace does not have a trace function.")

        # IMPORTANT: Replace the "tXX_vectorized" variables referenced by trace_vectorized.py
        if cls.vmodels is not None:
            if hasattr(cls.vmodels, "t89"):
                cls.tv.t89_vectorized = cls.vmodels.t89
            if hasattr(cls.vmodels, "t96"):
                cls.tv.t96_vectorized = cls.vmodels.t96
            if hasattr(cls.vmodels, "t01"):
                cls.tv.t01_vectorized = cls.vmodels.t01
            if hasattr(cls.vmodels, "t04"):
                cls.tv.t04_vectorized = cls.vmodels.t04

    def _run_scalar(self, xi, yi, zi, *, dir, rlim, r0, parmod, exname, inname, maxloop):
        gp = self.gp
        xf, yf, zf, xx, yy, zz = gp.trace(
            float(xi), float(yi), float(zi), float(dir),
            rlim=rlim, r0=r0,
            parmod=parmod,
            exname=exname, inname=inname,
            maxloop=maxloop
        )
        steps = len(xx) - 1
        status = infer_scalar_status(xf, yf, zf, r0=r0, rlim=rlim, steps=steps, maxloop=maxloop)
        return np.array([xf, yf, zf], dtype=np.float64), status

    def _run_vectorized(self, xi, yi, zi, *, dir, rlim, r0, parmod, exname, inname, maxloop):
        tv = self.tv
        xf, yf, zf, status = tv.trace(
            np.asarray(xi, dtype=np.float64),
            np.asarray(yi, dtype=np.float64),
            np.asarray(zi, dtype=np.float64),
            dir=float(dir), rlim=rlim, r0=r0,
            parmod=parmod, exname=exname, inname=inname,
            maxloop=maxloop,
            return_full_path=False, strict_scalar_models=False
        )
        out = np.stack([xf, yf, zf], axis=-1).astype(np.float64)
        return out, status.astype(np.int32)

    def _check_suite(self, exname: str, inname: str):
        if getattr(self, "skip_reason", None):
            self.skipTest(self.skip_reason)

        if self.vmodels is None or not hasattr(self.vmodels, exname.lower()):
            self.skipTest(f"{exname} vectorized not available in mageometry.geopack.vectorized.models")

        # For igrf, verify that vectorized internal model is available
        if inname.lower() == "igrf":
            try:
                from mageometry.geopack.vectorized.igrf import igrf_gsm as igrf_gsm_vectorized  # noqa: F401
            except Exception as e:
                self.skipTest(f"igrf_gsm_vectorized not available: {e}")

        rlim, r0, maxloop = 10.0, 1.0, 800
        parmod = parmod_for(exname)

        # Test points (small set)
        points = np.array([
            [2.0,  0.1,  0.0],
            [3.0,  1.0,  0.5],
            [4.5, -1.2,  0.8],
            [6.0,  0.5, -0.3],
            [8.0, -0.2,  0.1],
        ], dtype=np.float64)

        for dir in (+1.0, -1.0):
            xi, yi, zi = points[:, 0], points[:, 1], points[:, 2]

            vec_xyz, vec_status = self._run_vectorized(
                xi, yi, zi,
                dir=dir, rlim=rlim, r0=r0,
                parmod=parmod, exname=exname, inname=inname, maxloop=maxloop
            )

            for i in range(len(points)):
                where = f"(ex={exname}, in={inname}, dir={dir}, i={i}, p={points[i].tolist()})"
                with self.subTest(where=where):
                    sc_xyz, sc_status = self._run_scalar(
                        xi[i], yi[i], zi[i],
                        dir=dir, rlim=rlim, r0=r0,
                        parmod=parmod, exname=exname, inname=inname, maxloop=maxloop
                    )

                    # Require status match (can be relaxed if needed)
                    self.assertEqual(int(vec_status[i]), int(sc_status), f"status mismatch {where}")

                    np.testing.assert_allclose(
                        vec_xyz[i], sc_xyz,
                        rtol=TRACE_RTOL, atol=TRACE_ATOL_RE,
                        err_msg=f"endpoint mismatch {where} (atol={TRACE_ATOL_RE}Re)"
                    )

    def test_trace_external_models_dipole(self):
        for exname in ("t89", "t96", "t01", "t04"):
            with self.subTest(exname=exname):
                self._check_suite(exname=exname, inname="dipole")

    def test_trace_external_models_igrf(self):
        for exname in ("t89", "t96", "t01", "t04"):
            with self.subTest(exname=exname):
                self._check_suite(exname=exname, inname="igrf")


if __name__ == "__main__":
    unittest.main(verbosity=2)
