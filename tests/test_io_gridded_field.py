"""
Tests for mageometry.io: GriddedField interpolation and the XDMF/HDF5 readers.

A synthetic dipole field sampled on a grid serves as ground truth: the
interpolated field must reproduce the analytic values, and the geometry
functions applied to the interpolated field must reproduce the analytic
equatorial curvature 3/r.
"""

import os
import tempfile
import unittest

import numpy as np

from mageometry import GriddedField, field_line_curvature

try:
    import h5py
    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False


def dipole_b(x, y, z, m=1.0):
    """Analytic dipole field with moment m along +z (arbitrary units)."""
    r2 = x**2 + y**2 + z**2
    r5 = r2**2.5
    bx = 3.0 * m * x * z / r5
    by = 3.0 * m * y * z / r5
    bz = m * (3.0 * z**2 - r2) / r5
    return bx, by, bz


def write_xdmf_h5(dirpath, basename, origin, spacing, bx, by, bz,
                  datasets=('BX', 'BY', 'BZ')):
    """Write (nx, ny, nz) arrays as an XDMF + HDF5 pair in the on-disk
    convention used by MHD codes: big-endian float32 heavy data stored
    (nz, ny, nx), ZYX-ordered XDMF dims/origin/spacing.

    Returns (xmf_path, h5_path).
    """
    nx, ny, nz = bx.shape
    h5_path = os.path.join(dirpath, basename + '.h5')
    with h5py.File(h5_path, 'w') as f:
        for name, arr in zip(datasets, (bx, by, bz)):
            f.create_dataset(name, data=arr.transpose(2, 1, 0).astype('>f4'))

    dims_zyx = f"{nz} {ny} {nx}"
    origin_zyx = f"{origin[2]} {origin[1]} {origin[0]}"
    spacing_zyx = f"{spacing[2]} {spacing[1]} {spacing[0]}"
    attrs = "\n".join(
        f'<Attribute Name="{name}" AttributeType="Scalar" Center="Node">'
        f'<DataItem Dimensions="{dims_zyx}" NumberType="Float" Precision="4" '
        f'Format="HDF">{basename}.h5:/{name}</DataItem></Attribute>'
        for name in datasets
    )
    xmf_path = os.path.join(dirpath, basename + '.xmf')
    with open(xmf_path, 'w') as f:
        f.write(f"""<?xml version="1.0" ?>
<Xdmf Version="2.0"><Domain>
<Grid Name="Structured Grid" GridType="Uniform">
<Topology TopologyType="3DCORECTMesh" NumberOfElements="{dims_zyx}"/>
<Geometry GeometryType="ORIGIN_DXDYDZ">
<DataItem Name="Origin" Dimensions="3" NumberType="Float" Format="XML">{origin_zyx}</DataItem>
<DataItem Name="Spacing" Dimensions="3" NumberType="Float" Format="XML">{spacing_zyx}</DataItem>
</Geometry>
{attrs}
</Grid></Domain></Xdmf>
""")
    return xmf_path, h5_path


def make_dipole_grid(lo=-8.0, hi=8.0, n=81):
    ax = np.linspace(lo, hi, n)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing='ij')
    bx, by, bz = dipole_b(X, Y, Z)
    # Zero out the region near the singularity at the origin so that the grid
    # holds only finite, moderate values (required for spline interpolation).
    # Tests only evaluate at r >= 4, far from this clipped core.
    core = X**2 + Y**2 + Z**2 < 2.0**2
    for comp in (bx, by, bz):
        comp[core] = 0.0
    return ax, bx, by, bz


class TestGriddedField(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ax, bx, by, bz = make_dipole_grid()
        cls.grid = GriddedField(ax, ax, ax, bx, by, bz)

    def test_validation_rejects_bad_input(self):
        ax = np.linspace(0, 1, 5)
        good = np.zeros((5, 5, 5))
        with self.assertRaises(ValueError):
            GriddedField(ax[::-1], ax, ax, good, good, good)  # decreasing axis
        with self.assertRaises(ValueError):
            GriddedField(ax, ax, ax, good, good, np.zeros((5, 5, 4)))  # bad shape

    def test_interpolation_matches_analytic(self):
        field = self.grid.field(method='linear')
        rng = np.random.default_rng(0)
        # off-grid points, away from the origin where the dipole diverges
        x = rng.uniform(3, 7, 50) * rng.choice([-1, 1], 50)
        y = rng.uniform(-5, 5, 50)
        z = rng.uniform(-5, 5, 50)
        bx, by, bz = field(x, y, z)
        bx_a, by_a, bz_a = dipole_b(x, y, z)
        b_mag = np.sqrt(bx_a**2 + by_a**2 + bz_a**2)
        for got, want in ((bx, bx_a), (by, by_a), (bz, bz_a)):
            np.testing.assert_allclose(got, want, atol=2e-2 * b_mag.max(), rtol=0.05)

    def test_scalar_input_scalar_output(self):
        field = self.grid.field()
        b = field(5.0, 0.0, 0.0)
        self.assertEqual(len(b), 3)
        for comp in b:
            self.assertIsInstance(comp, float)

    def test_out_of_bounds_fill(self):
        field = self.grid.field(fill_value=np.nan)
        bx, by, bz = field(100.0, 0.0, 0.0)
        self.assertTrue(np.isnan(bx) and np.isnan(by) and np.isnan(bz))
        field0 = self.grid.field(fill_value=0.0)
        self.assertEqual(field0(100.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    def test_curvature_matches_analytic_dipole(self):
        """Equatorial field line curvature of a dipole is 3/r (grid units)."""
        field = self.grid.field(method='cubic')
        r = np.array([4.0, 5.0, 6.0])
        zeros = np.zeros_like(r)
        kappa = field_line_curvature(field, zeros, r, zeros, delta=0.2)
        np.testing.assert_allclose(kappa, 3.0 / r, rtol=0.05)

    def test_views_share_memory(self):
        self.assertTrue(np.shares_memory(self.grid.bx, self.grid.b))


@unittest.skipUnless(HAVE_H5PY, "h5py not installed")
class TestXdmfHdf5Readers(unittest.TestCase):
    """Round-trip through files written in the XDMF/HDF5 ZYX convention."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        nx, ny, nz = 6, 5, 4
        cls.origin = (-1.0, 0.0, 2.0)     # (x0, y0, z0)
        cls.spacing = (0.5, 1.0, 1.5)
        x = cls.origin[0] + cls.spacing[0] * np.arange(nx)
        y = cls.origin[1] + cls.spacing[1] * np.arange(ny)
        z = cls.origin[2] + cls.spacing[2] * np.arange(nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        # distinct, position-dependent components to catch axis-order mistakes
        cls.bx = (1.0 * X + 10.0 * Y + 100.0 * Z).astype(np.float32)
        cls.by = (2.0 * X - 3.0 * Y + 7.0 * Z).astype(np.float32)
        cls.bz = (-5.0 * X + 4.0 * Y - 1.0 * Z).astype(np.float32)

        cls.xmf_path, cls.h5_path = write_xdmf_h5(
            cls.tmp.name, 'mini', cls.origin, cls.spacing, cls.bx, cls.by, cls.bz
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _check(self, grid):
        self.assertEqual(grid.shape, (6, 5, 4))
        np.testing.assert_allclose(grid.x, self.origin[0] + self.spacing[0] * np.arange(6))
        np.testing.assert_allclose(grid.z, self.origin[2] + self.spacing[2] * np.arange(4))
        np.testing.assert_array_equal(grid.bx, self.bx)
        np.testing.assert_array_equal(grid.by, self.by)
        np.testing.assert_array_equal(grid.bz, self.bz)
        # interpolation at a grid node must return the exact node value
        bx, by, bz = grid.field()(grid.x[2], grid.y[1], grid.z[3])
        np.testing.assert_allclose((bx, by, bz),
                                   (self.bx[2, 1, 3], self.by[2, 1, 3], self.bz[2, 1, 3]),
                                   rtol=1e-6)

    def test_load_xdmf(self):
        from mageometry import load_xdmf
        self._check(load_xdmf(self.xmf_path))

    def test_load_xdmf_with_h5_override(self):
        from mageometry import load_xdmf
        self._check(load_xdmf(self.xmf_path, h5_file=self.h5_path))

    def test_load_hdf5_direct(self):
        from mageometry import load_hdf5
        self._check(load_hdf5(self.h5_path, origin=self.origin, spacing=self.spacing))

    def test_missing_dataset_raises(self):
        from mageometry import load_hdf5
        with self.assertRaises(KeyError):
            load_hdf5(self.h5_path, datasets=('NOPE', 'BY', 'BZ'),
                      origin=self.origin, spacing=self.spacing)


@unittest.skipUnless(HAVE_H5PY, "h5py not installed")
class TestTsyganenkoFileRoundtrip(unittest.TestCase):
    """End-to-end validation of the file pipeline against a known field.

    A T96+dipole field is sampled on a grid, written to an XDMF/HDF5 pair in
    the on-disk convention of MHD outputs, loaded back through `load_xdmf`,
    and interpolated. Because the model can also be evaluated directly at any
    point, it provides exact ground truth for both the field values and the
    geometry quantities computed through the file-based field.
    """

    DELTA = 0.25  # grid spacing (Re) and finite-difference step

    @classmethod
    def setUpClass(cls):
        from mageometry import geopack, geopack_field, load_xdmf

        ps = geopack.recalc(100)
        parmod = np.array([2.0, -20.0, 0.0, -5.0, 0, 0, 0, 0, 0, 0])
        # staticmethod: keep the callable unbound when accessed via self
        cls.direct = staticmethod(geopack_field('t96', 'dip', parmod, ps))

        # Nightside block, away from the model's inner region
        d = cls.DELTA
        x = np.arange(-9.0, -4.0 + d / 2, d)
        y = np.arange(-2.5, 2.5 + d / 2, d)
        z = np.arange(-2.5, 2.5 + d / 2, d)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        bx, by, bz = cls.direct(X.ravel(), Y.ravel(), Z.ravel())

        cls.tmp = tempfile.TemporaryDirectory()
        xmf_path, _ = write_xdmf_h5(
            cls.tmp.name, 't96block',
            origin=(x[0], y[0], z[0]), spacing=(d, d, d),
            bx=bx.reshape(X.shape).astype(np.float32),
            by=by.reshape(X.shape).astype(np.float32),
            bz=bz.reshape(X.shape).astype(np.float32),
        )
        cls.grid = load_xdmf(xmf_path)
        cls.axes = (x, y, z)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_axes_roundtrip(self):
        for got, want in zip((self.grid.x, self.grid.y, self.grid.z), self.axes):
            np.testing.assert_allclose(got, want, atol=1e-6)

    def test_interpolated_field_matches_model(self):
        # calibrated: linear interp on a 0.25 Re grid gives max rel err ~3e-3
        field = self.grid.field(method='linear')
        rng = np.random.default_rng(1)
        n = 200
        x = rng.uniform(-8.5, -4.5, n)
        y = rng.uniform(-2.0, 2.0, n)
        z = rng.uniform(-2.0, 2.0, n)
        b_interp = np.array(field(x, y, z))
        b_direct = np.array(self.direct(x, y, z))
        rel_err = (np.linalg.norm(b_interp - b_direct, axis=0)
                   / np.linalg.norm(b_direct, axis=0))
        self.assertLess(rel_err.max(), 1e-2)

    def _curvature_case(self, method, rtol):
        field = self.grid.field(method=method)
        x = np.array([-7.0, -6.0, -5.5])
        y = np.array([0.5, -1.0, 1.5])
        z = np.array([1.0, 1.5, -1.0])
        kappa_file = field_line_curvature(field, x, y, z, delta=self.DELTA)
        kappa_direct = field_line_curvature(self.direct, x, y, z, delta=self.DELTA)
        np.testing.assert_allclose(kappa_file, kappa_direct, rtol=rtol)

    def test_curvature_through_file_linear(self):
        # calibrated: ~0.5% observed
        self._curvature_case('linear', rtol=2e-2)

    def test_curvature_through_file_cubic(self):
        # calibrated: ~0.1% observed
        self._curvature_case('cubic', rtol=5e-3)

    def test_frenet_frame_through_file(self):
        """The file-based field must reproduce the direct-model frame,
        including the geometry module's validity masking (the normal is NaN
        where it cannot be defined reliably — that masking applies
        identically to both field sources and is not an io property)."""
        from mageometry import field_line_frenet_frame, verify_unit_vectors
        field = self.grid.field(method='cubic')
        x = np.array([-7.0, -6.0, -5.5, -6.5, -8.0, -5.0])
        y = np.array([0.5, -1.0, 1.5, 0.0, 1.0, -0.5])
        z = np.array([1.0, 1.5, -1.0, 1.0, -1.5, 1.5])
        frame_d = field_line_frenet_frame(self.direct, x, y, z, delta=self.DELTA)
        frame_f = field_line_frenet_frame(field, x, y, z, delta=self.DELTA)

        valid_d = np.isfinite(frame_d[3])
        valid_f = np.isfinite(frame_f[3])
        np.testing.assert_array_equal(valid_f, valid_d)
        self.assertTrue(np.all(valid_d), "all test points should be valid at delta=0.25")

        # tangent is defined everywhere; normal/binormal where valid
        for label, i, mask in [('T', 0, slice(None)), ('n', 3, valid_d),
                               ('b', 6, valid_d)]:
            dot = sum(frame_d[i + k][mask] * frame_f[i + k][mask] for k in range(3))
            np.testing.assert_allclose(dot, 1.0, atol=1e-3, err_msg=label)

        # The frame is orthonormal by construction (the normal is the
        # projection of dT/ds perpendicular to T), independent of the data.
        errors = verify_unit_vectors(*(comp[valid_f] for comp in frame_f[:9]))
        for name, err in errors.items():
            self.assertLess(np.max(np.abs(err)), 1e-12, msg=name)


if __name__ == '__main__':
    unittest.main()
