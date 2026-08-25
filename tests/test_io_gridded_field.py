"""
Tests for mageometry.io: GriddedField interpolation and the XDMF/HDF5 readers.

A synthetic dipole field sampled on a grid serves as ground truth: the
interpolated field must reproduce the analytic values, and the geometry
functions applied to the interpolated field must reproduce the analytic
equatorial curvature 3/r.
"""

import json
import os
import tempfile
import unittest

import numpy as np

from mageometry import GriddedField, field_line_curvature
from mageometry.io import region_slices

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


def _write_h5(h5_path, arrays, datasets):
    with h5py.File(h5_path, 'w') as f:
        for name, arr in zip(datasets, arrays):
            f.create_dataset(name, data=arr.transpose(2, 1, 0).astype('>f4'))


def _grid_xml(basename, origin, spacing, shape, datasets, center='Node',
              time=None, name="Structured Grid"):
    """XML for one uniform <Grid>; shape is the (nx, ny, nz) *data* shape.

    For cell-centered data the topology declares one more node per axis.
    """
    nx, ny, nz = shape
    if center == 'Cell':
        nx, ny, nz = nx + 1, ny + 1, nz + 1
    dims_zyx = f"{nz} {ny} {nx}"
    data_zyx = f"{shape[2]} {shape[1]} {shape[0]}"
    origin_zyx = f"{origin[2]} {origin[1]} {origin[0]}"
    spacing_zyx = f"{spacing[2]} {spacing[1]} {spacing[0]}"
    attrs = "\n".join(
        f'<Attribute Name="{n}" AttributeType="Scalar" Center="{center}">'
        f'<DataItem Dimensions="{data_zyx}" NumberType="Float" Precision="4" '
        f'Format="HDF">{basename}.h5:/{n}</DataItem></Attribute>'
        for n in datasets
    )
    time_xml = f'<Time Value="{time}"/>\n' if time is not None else ''
    return f"""<Grid Name="{name}" GridType="Uniform">
{time_xml}<Topology TopologyType="3DCORECTMesh" NumberOfElements="{dims_zyx}"/>
<Geometry GeometryType="ORIGIN_DXDYDZ">
<DataItem Name="Origin" Dimensions="3" NumberType="Float" Format="XML">{origin_zyx}</DataItem>
<DataItem Name="Spacing" Dimensions="3" NumberType="Float" Format="XML">{spacing_zyx}</DataItem>
</Geometry>
{attrs}
</Grid>"""


def write_xdmf_h5(dirpath, basename, origin, spacing, bx, by, bz,
                  datasets=('BX', 'BY', 'BZ'), center='Node', time=None):
    """Write (nx, ny, nz) arrays as an XDMF + HDF5 pair in the on-disk
    convention used by MHD codes: big-endian float32 heavy data stored
    (nz, ny, nx), ZYX-ordered XDMF dims/origin/spacing.

    Returns (xmf_path, h5_path).
    """
    h5_path = os.path.join(dirpath, basename + '.h5')
    _write_h5(h5_path, (bx, by, bz), datasets)
    xmf_path = os.path.join(dirpath, basename + '.xmf')
    with open(xmf_path, 'w') as f:
        f.write('<?xml version="1.0" ?>\n<Xdmf Version="2.0"><Domain>\n'
                + _grid_xml(basename, origin, spacing, bx.shape, datasets, center, time)
                + '\n</Domain></Xdmf>\n')
    return xmf_path, h5_path


def write_xdmf_collection(dirpath, basename, origin, spacing, steps,
                          datasets=('BX', 'BY', 'BZ')):
    """Write a temporal collection: one XDMF with one <Grid> per (time, bx, by, bz)."""
    grids = []
    for k, (t, bx, by, bz) in enumerate(steps):
        step_base = f"{basename}_{k:03d}"
        _write_h5(os.path.join(dirpath, step_base + '.h5'), (bx, by, bz), datasets)
        grids.append(_grid_xml(step_base, origin, spacing, bx.shape, datasets, time=t,
                               name=f"step {k}"))
    xmf_path = os.path.join(dirpath, basename + '.xmf')
    with open(xmf_path, 'w') as f:
        f.write('<?xml version="1.0" ?>\n<Xdmf Version="2.0"><Domain>\n'
                '<Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">\n'
                + "\n".join(grids) + '\n</Grid></Domain></Xdmf>\n')
    return xmf_path


def write_xmf_series(dirpath, basename, origin, spacing, steps,
                     datasets=('BX', 'BY', 'BZ')):
    """Write a ParaView .xmf.series index plus one single-grid XDMF per step."""
    entries = []
    for k, (t, bx, by, bz) in enumerate(steps):
        step_base = f"{basename}_{k:03d}"
        write_xdmf_h5(dirpath, step_base, origin, spacing, bx, by, bz, datasets)
        entries.append({"name": step_base + ".xmf", "time": t})
    series_path = os.path.join(dirpath, basename + '.xmf.series')
    with open(series_path, 'w') as f:
        json.dump({"file-series-version": "1.0", "files": entries}, f)
    return series_path


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


class TestRegionAndSubvolume(unittest.TestCase):

    def test_region_slices(self):
        ax = np.linspace(0.0, 10.0, 11)
        sx, sy, sz = region_slices((ax, ax, ax), ((2.0, 5.0), None, (7.5, 10.0)), stride=1)
        np.testing.assert_array_equal(ax[sx], [2, 3, 4, 5])
        self.assertEqual(ax[sy].size, 11)
        np.testing.assert_array_equal(ax[sz], [8, 9, 10])
        (sx,) = region_slices((ax,), ((0.0, 10.0),), stride=(3,))
        np.testing.assert_array_equal(ax[sx], [0, 3, 6, 9])
        with self.assertRaises(ValueError):
            region_slices((ax,), ((4.2, 4.8),))   # no node inside
        with self.assertRaises(ValueError):
            region_slices((ax,), ((2.0, 5.0), None))  # wrong length

    def test_subvolume(self):
        ax, bx, by, bz = make_dipole_grid(n=41)
        grid = GriddedField(ax, ax, ax, bx, by, bz, metadata={'source': 'mem'})
        sub = grid.subvolume(((-4.0, 4.0), None, (0.0, 8.0)), stride=(2, 1, 1))
        np.testing.assert_array_equal(sub.x, ax[(ax >= -4) & (ax <= 4)][::2])
        np.testing.assert_array_equal(sub.y, ax)
        np.testing.assert_array_equal(sub.z, ax[ax >= 0])
        # values are the corresponding slice of the parent
        i0 = int(np.searchsorted(ax, -4.0)); k0 = int(np.searchsorted(ax, 0.0))
        np.testing.assert_array_equal(sub.bx, grid.bx[i0:i0 + sub.x.size * 2:2, :, k0:])
        self.assertEqual(sub.metadata['source'], 'mem')
        # copy, not a view
        sub.b[...] = 0
        self.assertNotEqual(np.abs(grid.bx).max(), 0)


@unittest.skipUnless(HAVE_H5PY, "h5py not installed")
class TestReaderRegionStrideAndCentering(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.ax, bx, by, bz = make_dipole_grid(n=33)
        cls.arrays = (bx, by, bz)
        cls.origin = (cls.ax[0],) * 3
        cls.spacing = (cls.ax[1] - cls.ax[0],) * 3
        cls.xmf, cls.h5 = write_xdmf_h5(cls.tmp.name, 'node', cls.origin, cls.spacing, bx, by, bz)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_region_and_stride_match_full_load(self):
        from mageometry import load_xdmf, load_hdf5
        full = load_xdmf(self.xmf)
        region = ((-4.0, 2.0), None, (-8.0, 0.0))
        part = load_xdmf(self.xmf, region=region, stride=(1, 2, 1))
        sub = full.subvolume(region, stride=(1, 2, 1))
        for a, b in ((part.x, sub.x), (part.y, sub.y), (part.z, sub.z)):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(part.b, sub.b)
        self.assertEqual(part.b.dtype, np.float32)

        part_h5 = load_hdf5(self.h5, origin=self.origin, spacing=self.spacing,
                            region=region, stride=(1, 2, 1))
        np.testing.assert_array_equal(part_h5.b, sub.b)
        np.testing.assert_array_equal(part_h5.x, sub.x)

    def test_stride_only(self):
        from mageometry import load_xdmf
        coarse = load_xdmf(self.xmf, stride=4)
        np.testing.assert_array_equal(coarse.x, self.ax[::4])
        np.testing.assert_array_equal(coarse.bx, self.arrays[0][::4, ::4, ::4].astype(np.float32))

    def test_cell_centered(self):
        from mageometry import load_xdmf
        # Cell-centered data: values at (i + 1/2) * spacing; the topology
        # declares n + 1 nodes per axis.
        bx, by, bz = self.arrays
        xmf, _ = write_xdmf_h5(self.tmp.name, 'cell', self.origin, self.spacing,
                               bx, by, bz, center='Cell')
        grid = load_xdmf(xmf)
        self.assertEqual(grid.shape, bx.shape)
        dx = self.spacing[0]
        np.testing.assert_allclose(grid.x, self.ax + 0.5 * dx)
        np.testing.assert_array_equal(grid.bx, bx.astype(np.float32))
        self.assertEqual(grid.metadata['center'], 'cell')
        # region selection uses the cell-center coordinates
        part = load_xdmf(xmf, region=((0.0, 3.0), None, None))
        self.assertTrue(np.all(part.x >= 0.0) and np.all(part.x <= 3.0))

    def test_mixed_centering_rejected(self):
        from mageometry import load_xdmf
        with open(self.xmf) as f:
            xml = f.read()
        xml = xml.replace('Name="BZ" AttributeType="Scalar" Center="Node"',
                          'Name="BZ" AttributeType="Scalar" Center="Cell"', 1)
        path = os.path.join(self.tmp.name, 'mixed.xmf')
        with open(path, 'w') as f:
            f.write(xml)
        with self.assertRaises(ValueError):
            load_xdmf(path)

    def test_time_metadata(self):
        from mageometry import load_xdmf
        bx, by, bz = self.arrays
        xmf, _ = write_xdmf_h5(self.tmp.name, 'timed', self.origin, self.spacing,
                               bx, by, bz, time=12.5)
        self.assertEqual(load_xdmf(xmf).metadata['time'], 12.5)


@unittest.skipUnless(HAVE_H5PY, "h5py not installed")
class TestXdmfSeries(unittest.TestCase):
    """Time series: the field at step k is the dipole scaled by (1 + t_k)."""

    TIMES = [0.0, 0.5, 2.0]

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        ax, bx, by, bz = make_dipole_grid(n=17)
        cls.ax = ax
        cls.base = (bx, by, bz)
        origin = (ax[0],) * 3
        spacing = (ax[1] - ax[0],) * 3
        steps = [(t, bx * (1 + t), by * (1 + t), bz * (1 + t)) for t in cls.TIMES]
        cls.collection = write_xdmf_collection(cls.tmp.name, 'coll', origin, spacing, steps)
        cls.series = write_xmf_series(cls.tmp.name, 'run', origin, spacing, steps)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _check(self, series):
        self.assertEqual(len(series), 3)
        np.testing.assert_array_equal(series.times, self.TIMES)
        for k, grid in enumerate(series):
            np.testing.assert_array_equal(grid.x, self.ax)
            np.testing.assert_allclose(grid.bz, (self.base[2] * (1 + self.TIMES[k])).astype(np.float32),
                                       rtol=1e-6)
            self.assertEqual(grid.metadata.get('time'), self.TIMES[k])
        self.assertEqual(series.index_at(0.6), 1)
        self.assertEqual(series.index_at(100.0), 2)
        np.testing.assert_allclose(series.at(1.9).bx, (self.base[0] * 3.0).astype(np.float32), rtol=1e-6)
        np.testing.assert_allclose(series[-1].bx, series[2].bx)
        with self.assertRaises(IndexError):
            series[3]
        # slicing gives a shorter series
        tail = series[1:]
        self.assertEqual(len(tail), 2)
        np.testing.assert_array_equal(tail.times, self.TIMES[1:])

    def test_temporal_collection(self):
        from mageometry import load_xdmf_series, load_xdmf
        self._check(load_xdmf_series(self.collection))
        with self.assertRaises(ValueError):
            load_xdmf(self.collection)   # must point users to the series reader

    def test_xmf_series_index(self):
        from mageometry import load_xdmf_series
        self._check(load_xdmf_series(self.series))

    def test_series_region_passthrough(self):
        from mageometry import load_xdmf_series
        series = load_xdmf_series(self.series, region=((0.0, 8.0), None, None), stride=2)
        grid = series[0]
        self.assertTrue(np.all(grid.x >= 0.0))
        np.testing.assert_array_equal(grid.x, self.ax[self.ax >= 0.0][::2])
        np.testing.assert_array_equal(grid.y, self.ax[::2])

    def test_single_grid_is_not_a_series(self):
        from mageometry import load_xdmf_series
        single, _ = write_xdmf_h5(self.tmp.name, 'single', (self.ax[0],) * 3,
                                  (self.ax[1] - self.ax[0],) * 3, *self.base)
        with self.assertRaises(ValueError):
            load_xdmf_series(single)


class TestBinaryHelpers(unittest.TestCase):

    def _write_sequential(self, path, records, dtype, marker='i4'):
        m = np.dtype(marker).newbyteorder(np.dtype(dtype).byteorder or '=')
        with open(path, 'wb') as f:
            for rec in records:
                payload = np.asarray(rec).astype(dtype).tobytes()
                f.write(np.array(len(payload), dtype=m).tobytes())
                f.write(payload)
                f.write(np.array(len(payload), dtype=m).tobytes())

    def test_fortran_records_roundtrip(self):
        from mageometry.io import read_fortran_records, iter_fortran_records
        rng = np.random.default_rng(0)
        recs = [rng.standard_normal(12), rng.standard_normal(30), rng.standard_normal(5)]
        with tempfile.TemporaryDirectory() as d:
            for dtype, marker in (('>f4', 'i4'), ('<f8', 'i4'), ('<f4', 'i8')):
                path = os.path.join(d, 'seq.bin')
                self._write_sequential(path, recs, dtype, marker)
                kw = {'marker_dtype': '<i8'} if marker == 'i8' else {}
                got = read_fortran_records(path, dtype=dtype, **kw)
                self.assertEqual(len(got), 3)
                for g, r in zip(got, recs):
                    self.assertEqual(g.dtype.byteorder, '=')
                    np.testing.assert_allclose(g, r.astype(dtype), rtol=1e-6)
                # skip / count, and the iterator
                got = read_fortran_records(path, dtype=dtype, skip=1, count=1, **kw)
                self.assertEqual(len(got), 1)
                self.assertEqual(got[0].size, 30)
                self.assertEqual(sum(1 for _ in iter_fortran_records(path, dtype, **kw)), 3)
            # wrong byte order is detected from the markers
            path = os.path.join(d, 'seq.bin')
            self._write_sequential(path, recs, '>f4')
            with self.assertRaises(ValueError):
                read_fortran_records(path, dtype='<f4')

    def test_field_series_from_files(self):
        from mageometry.io import FieldSeries
        ax = np.linspace(-1, 1, 5)
        calls = []

        def loader(path, scale=1.0):
            calls.append(path)
            k = int(path[-1])
            z = np.zeros((5, 5, 5))
            return GriddedField(ax, ax, ax, z + k * scale, z, z, metadata={'path': path})

        series = FieldSeries.from_files(['step0', 'step1', 'step2'], loader,
                                        times=[0.0, 1.0, 2.5], scale=2.0)
        self.assertEqual(len(series), 3)
        self.assertEqual(calls, [])                # lazy: nothing read yet
        np.testing.assert_array_equal(series.times, [0.0, 1.0, 2.5])
        self.assertEqual(series.at(2.0).bx[0, 0, 0], 4.0)
        self.assertEqual(calls, ['step2'])
        self.assertEqual([g.metadata['path'] for g in series[1:]], ['step1', 'step2'])
        with self.assertRaises(ValueError):
            FieldSeries.from_files(['a', 'b'], loader, times=[0.0])
        untimed = FieldSeries.from_files(['step0'], loader)
        self.assertTrue(np.isnan(untimed.times[0]))
        with self.assertRaises(ValueError):
            untimed.at(0.0)


class TestDivergenceCheck(unittest.TestCase):

    def test_divergence_detects_permutations(self):
        ax, bx, by, bz = make_dipole_grid(n=41)
        grid = GriddedField(ax, ax, ax, bx, by, bz)
        X, Y, Z = np.meshgrid(ax, ax, ax, indexing='ij')
        outside = X**2 + Y**2 + Z**2 > 3.0**2
        # measured: correct ~1e-3; permuted/transposed/sign-flipped 0.09-0.14
        good = np.nanmedian(grid.divergence()[outside])
        self.assertLess(good, 0.01)
        # components permuted -> not divergence-free
        bad = GriddedField(ax, ax, ax, by, bz, bx)
        self.assertGreater(np.nanmedian(bad.divergence()[outside]), 0.05)
        # axes transposed (data stored (nz, ny, nx) but declared (nx, ny, nz))
        bad = GriddedField(ax, ax, ax, bx.transpose(2, 1, 0), by.transpose(2, 1, 0), bz.transpose(2, 1, 0))
        self.assertGreater(np.nanmedian(bad.divergence()[outside]), 0.05)
        # one component sign-flipped
        bad = GriddedField(ax, ax, ax, bx, by, -bz)
        self.assertGreater(np.nanmedian(bad.divergence()[outside]), 0.05)
        # absolute divergence has the right shape and units
        self.assertEqual(grid.divergence(relative=False).shape, grid.shape)


if __name__ == '__main__':
    unittest.main()
