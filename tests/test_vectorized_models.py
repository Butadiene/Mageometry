# tests/test_vectorized_models.py
import os
import sys
import unittest
import datetime
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 許容誤差（環境変数で調整可能）
FIELD_RTOL = float(os.environ.get("GEOPACK_FIELD_RTOL", "1e-10"))
FIELD_ATOL = float(os.environ.get("GEOPACK_FIELD_ATOL", "1e-6"))  # nT


def ut_seconds(dt: datetime.datetime) -> float:
    epoch = datetime.datetime(1970, 1, 1)
    return (dt - epoch).total_seconds()


def make_points(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-10.0, 10.0, size=n)
    y = rng.uniform(-5.0, 5.0, size=n)
    z = rng.uniform(-5.0, 5.0, size=n)

    r = np.sqrt(x * x + y * y + z * z)
    mask = r < 1.2
    if np.any(mask):
        x[mask] += 2.0

    return x.astype(np.float64), y.astype(np.float64), z.astype(np.float64)


def eval_scalar_loop(fn, parmod, ps, x, y, z):
    bx = np.empty_like(x, dtype=np.float64)
    by = np.empty_like(y, dtype=np.float64)
    bz = np.empty_like(z, dtype=np.float64)
    for i in range(len(x)):
        bx[i], by[i], bz[i] = fn(parmod, ps, float(x[i]), float(y[i]), float(z[i]))
    return bx, by, bz


class TestVectorizedExternalModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import geopack.geopack as gp
        cls.gp = gp

        ut = ut_seconds(datetime.datetime(2020, 3, 20, 0, 0, 0))
        gp.recalc(ut)
        cls.ps = gp.psi

        # vectorized models は geopack.vectorized.models に居る前提
        try:
            from geopack.vectorized import models as vmodels
        except Exception as e:
            vmodels = None
        cls.vmodels = vmodels

    def _assert_vec_model_close(self, name, scalar_fn, vec_fn, parmod):
        x, y, z = make_points(n=300, seed=1)

        bx_s, by_s, bz_s = eval_scalar_loop(scalar_fn, parmod, self.ps, x, y, z)
        bx_v, by_v, bz_v = vec_fn(parmod, self.ps, x, y, z)

        np.testing.assert_allclose(bx_v, bx_s, rtol=FIELD_RTOL, atol=FIELD_ATOL, err_msg=f"{name}: bx mismatch")
        np.testing.assert_allclose(by_v, by_s, rtol=FIELD_RTOL, atol=FIELD_ATOL, err_msg=f"{name}: by mismatch")
        np.testing.assert_allclose(bz_v, bz_s, rtol=FIELD_RTOL, atol=FIELD_ATOL, err_msg=f"{name}: bz mismatch")

    def test_t89_vectorized(self):
        from geopack.models import t89

        if self.vmodels is None or not hasattr(self.vmodels, "t89"):
            self.skipTest("t89 vectorized not available in geopack.vectorized.models")

        parmod = 2
        self._assert_vec_model_close("t89", t89, self.vmodels.t89, parmod)

    def test_t96_vectorized(self):
        from geopack.models import t96

        if self.vmodels is None or not hasattr(self.vmodels, "t96"):
            self.skipTest("t96 vectorized not available in geopack.vectorized.models")

        parmod = np.array([2.0, -20.0, 0.0, -5.0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        self._assert_vec_model_close("t96", t96, self.vmodels.t96, parmod)

    def test_t01_vectorized(self):
        from geopack.models import t01

        if self.vmodels is None or not hasattr(self.vmodels, "t01"):
            self.skipTest("t01 vectorized not available in geopack.vectorized.models")

        parmod = np.array([2.0, -20.0, 0.0, -5.0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        self._assert_vec_model_close("t01", t01, self.vmodels.t01, parmod)

    def test_t04_vectorized(self):
        from geopack.models import t04

        if self.vmodels is None or not hasattr(self.vmodels, "t04"):
            self.skipTest("t04 vectorized not available in geopack.vectorized.models")

        parmod = np.array([2.0, -20.0, 0.0, -5.0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        self._assert_vec_model_close("t04", t04, self.vmodels.t04, parmod)


class TestVectorizedInternalModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import geopack.geopack as gp
        cls.gp = gp
        ut = ut_seconds(datetime.datetime(2020, 3, 20, 0, 0, 0))
        gp.recalc(ut)

    def test_igrf_gsm_vectorized(self):
        try:
            from geopack.vectorized.igrf import igrf_gsm as igrf_gsm_vectorized
        except Exception as e:
            self.skipTest(f"igrf_gsm_vectorized not available: {e}")

        gp = self.gp
        x, y, z = make_points(n=200, seed=2)

        bx_s = np.empty_like(x); by_s = np.empty_like(x); bz_s = np.empty_like(x)
        for i in range(len(x)):
            bx_s[i], by_s[i], bz_s[i] = gp.igrf_gsm(float(x[i]), float(y[i]), float(z[i]))

        bx_v, by_v, bz_v = igrf_gsm_vectorized(x, y, z)

        np.testing.assert_allclose(bx_v, bx_s, rtol=FIELD_RTOL, atol=FIELD_ATOL, err_msg="igrf: bx mismatch")
        np.testing.assert_allclose(by_v, by_s, rtol=FIELD_RTOL, atol=FIELD_ATOL, err_msg="igrf: by mismatch")
        np.testing.assert_allclose(bz_v, bz_s, rtol=FIELD_RTOL, atol=FIELD_ATOL, err_msg="igrf: bz mismatch")


if __name__ == "__main__":
    unittest.main(verbosity=2)
