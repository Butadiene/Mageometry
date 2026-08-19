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

        cls.h5_path = os.path.join(cls.tmp.name, 'mini.h5')
        with h5py.File(cls.h5_path, 'w') as f:
            for name, arr in (('BX', cls.bx), ('BY', cls.by), ('BZ', cls.bz)):
                # store as (nz, ny, nx) big-endian, like the MHD sample data
                f.create_dataset(name, data=arr.transpose(2, 1, 0).astype('>f4'))

        cls.xmf_path = os.path.join(cls.tmp.name, 'mini.xmf')
        dims_zyx = f"{nz} {ny} {nx}"
        origin_zyx = f"{cls.origin[2]} {cls.origin[1]} {cls.origin[0]}"
        spacing_zyx = f"{cls.spacing[2]} {cls.spacing[1]} {cls.spacing[0]}"
        attrs = "\n".join(
            f'<Attribute Name="{name}" AttributeType="Scalar" Center="Node">'
            f'<DataItem Dimensions="{dims_zyx}" NumberType="Float" Precision="4" '
            f'Format="HDF">mini.h5:/{name}</DataItem></Attribute>'
            for name in ('BX', 'BY', 'BZ')
        )
        with open(cls.xmf_path, 'w') as f:
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


if __name__ == '__main__':
    unittest.main()
