"""
Tests for mageometry.io.load_vtk (requires pyvista; skipped otherwise).

The round trip GriddedField -> viz3d.to_rectilinear_grid -> .vtr file ->
load_vtk proves that the two transposes (ij order <-> VTK x-fastest order)
are mutual inverses.
"""

import os
import tempfile
import unittest

import numpy as np

from mageometry import GriddedField, load_vtk

try:
    import pyvista as pv
    pv.OFF_SCREEN = True
    HAVE_PV = True
except ImportError:
    HAVE_PV = False


def make_field():
    x, y, z = np.arange(3.0), 1.0 + np.arange(4.0) * 2.0, np.arange(5.0) * 3.0
    i, j, k = np.meshgrid(np.arange(3), np.arange(4), np.arange(5), indexing='ij')
    bx = 100.0 * i + 10.0 * j + k
    return GriddedField(x, y, z, bx, bx + 0.25, bx - 0.25)


@unittest.skipUnless(HAVE_PV, "pyvista not installed")
class TestLoadVtk(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def test_rectilinear_round_trip(self):
        from mageometry import viz3d
        gf = make_field()
        path = self.path('field.vtr')
        viz3d.to_rectilinear_grid(gf, quantities=()).save(path)
        back = load_vtk(path)
        np.testing.assert_array_equal(back.x, gf.x)
        np.testing.assert_array_equal(back.y, gf.y)
        np.testing.assert_array_equal(back.z, gf.z)
        np.testing.assert_array_equal(back.b, gf.b)
        self.assertEqual(back.metadata['association'], 'point')

    def test_region_and_stride(self):
        from mageometry import viz3d
        gf = make_field()
        path = self.path('field.vtr')
        viz3d.to_rectilinear_grid(gf, quantities=()).save(path)
        back = load_vtk(path, region=(None, (1.0, 5.0), None), stride=(1, 1, 2))
        sub = gf.subvolume(region=(None, (1.0, 5.0), None), stride=(1, 1, 2))
        np.testing.assert_array_equal(back.y, sub.y)
        np.testing.assert_array_equal(back.z, sub.z)
        np.testing.assert_array_equal(back.b, sub.b)

    def test_imagedata_point_and_cell_data(self):
        gf = make_field()
        img = pv.ImageData(dimensions=(3, 4, 5), spacing=(1.0, 2.0, 3.0),
                           origin=(0.0, 1.0, 0.0))
        vec = np.column_stack([gf.b[..., k].ravel(order='F') for k in range(3)])
        img.point_data['B'] = vec
        # cell data on the same mesh: 2x3x4 cells, values encode cell indices
        i, j, k = np.meshgrid(np.arange(2), np.arange(3), np.arange(4), indexing='ij')
        cvals = (100.0 * i + 10.0 * j + k).astype(np.float64)
        img.cell_data['Bc'] = np.column_stack(
            [cvals.ravel(order='F') + m for m in range(3)])
        path = self.path('field.vti')
        img.save(path)

        back = load_vtk(path)
        np.testing.assert_array_equal(back.x, gf.x)
        np.testing.assert_array_equal(back.y, gf.y)
        np.testing.assert_array_equal(back.b, gf.b)

        cell = load_vtk(path, name='Bc')
        self.assertEqual(cell.shape, (2, 3, 4))
        np.testing.assert_array_equal(cell.x, [0.5, 1.5])          # cell centers
        np.testing.assert_array_equal(cell.y, [2.0, 4.0, 6.0])
        np.testing.assert_array_equal(cell.bx, cvals)
        np.testing.assert_array_equal(cell.bz, cvals + 2)
        self.assertEqual(cell.metadata['association'], 'cell')

    def test_three_scalar_arrays(self):
        gf = make_field()
        img = pv.ImageData(dimensions=(3, 4, 5), spacing=(1.0, 2.0, 3.0),
                           origin=(0.0, 1.0, 0.0))
        for key, comp in zip(('bx', 'by', 'bz'), (gf.bx, gf.by, gf.bz)):
            img.point_data[key] = comp.ravel(order='F')
        path = self.path('scalars.vti')
        img.save(path)
        back = load_vtk(path, name=('bx', 'by', 'bz'))
        np.testing.assert_array_equal(back.b, gf.b)

    def test_errors(self):
        img = pv.ImageData(dimensions=(3, 4, 5))
        img.point_data['scalar'] = np.zeros(3 * 4 * 5)
        path = self.path('bad.vti')
        img.save(path)
        with self.assertRaises(KeyError):
            load_vtk(path)                       # no 'B'
        with self.assertRaises(ValueError):
            load_vtk(path, name='scalar')        # not 3-component
        with self.assertRaises(ValueError):
            load_vtk(path, name=('a', 'b'))      # wrong name count
        # non-rectilinear mesh
        sphere = pv.Sphere()
        spath = self.path('sphere.vtp')
        sphere.save(spath)
        with self.assertRaises(ValueError):
            load_vtk(spath)


if __name__ == '__main__':
    unittest.main()
