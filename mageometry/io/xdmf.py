# mageometry/io/xdmf.py
"""
Readers for XDMF-described and plain-HDF5 gridded magnetic field data.

XDMF (https://xdmf.org) is an XML metadata format used by many simulation
codes (and readable by ParaView/VisIt) that describes grid geometry and points
to "heavy data" stored in HDF5. `load_xdmf` supports the common case of a
uniform structured grid (``3DCORECTMesh`` topology with ``ORIGIN_DXDYDZ``
geometry) with node-centered scalar attributes for the field components.

`load_hdf5` reads the HDF5 datasets directly when no XDMF file is available;
the grid geometry is then supplied by the caller.

Both readers return a `GriddedField`. HDF5 access requires the optional
`h5py` dependency (``pip install mageometry[io]``).

Axis-order convention: XDMF lists dimensions and Origin/DxDyDz values slowest
axis first (Z, Y, X for 3D), matching the C-order HDF5 dataset layout
``(nz, ny, nx)``. The readers transpose to the Mageometry convention
``(nx, ny, nz)``; use ``zyx_order=False`` if your data is already (nx, ny, nz).
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

from .gridded_field import GriddedField


def _require_h5py():
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "Reading HDF5 data requires the optional dependency h5py. "
            "Install it with: pip install h5py"
        ) from None
    return h5py


def _axes_from_origin_spacing(origin, spacing, shape):
    return tuple(
        o + s * np.arange(n, dtype=np.float64)
        for o, s, n in zip(origin, spacing, shape)
    )


def load_hdf5(path, datasets=('BX', 'BY', 'BZ'), origin=(0.0, 0.0, 0.0),
              spacing=(1.0, 1.0, 1.0), zyx_order=True, metadata=None):
    """
    Load field components from an HDF5 file on a uniform grid.

    Parameters
    ----------
    path : str
        HDF5 file path.
    datasets : tuple of str, optional
        Names of the (bx, by, bz) datasets. Default ('BX', 'BY', 'BZ').
    origin, spacing : tuple of float, optional
        Coordinate of the first grid node and grid step, per axis, in
        (x, y, z) order.
    zyx_order : bool, optional
        If True (default) the datasets are stored as ``(nz, ny, nx)`` (the
        usual C-order convention, as in XDMF) and are transposed to
        ``(nx, ny, nz)``. Set False if they are already (nx, ny, nz).
    metadata : dict, optional
        Extra provenance merged into ``GriddedField.metadata``.

    Returns
    -------
    GriddedField
    """
    h5py = _require_h5py()
    comps = []
    with h5py.File(path, 'r') as f:
        for name in datasets:
            if name not in f:
                raise KeyError(
                    f"Dataset {name!r} not found in {path!r}; "
                    f"available: {sorted(f.keys())}"
                )
            arr = f[name][()]
            if zyx_order:
                arr = arr.transpose(2, 1, 0)
            # native-endian copy (h5py may return big-endian data)
            comps.append(np.ascontiguousarray(arr, dtype=arr.dtype.newbyteorder('=')))

    x, y, z = _axes_from_origin_spacing(origin, spacing, comps[0].shape)
    meta = {'source': os.path.abspath(path), 'datasets': tuple(datasets)}
    if metadata:
        meta.update(metadata)
    return GriddedField(x, y, z, *comps, metadata=meta)


def _parse_xdmf_uniform_grid(path, components):
    """Parse an XDMF file; return (shape_xyz, origin_xyz, spacing_xyz, refs).

    refs maps component name -> (h5_filename, h5_dataset).
    """
    root = ET.parse(path).getroot()
    grid = root.find('.//Grid')
    if grid is None:
        raise ValueError(f"No <Grid> element found in {path!r}.")

    topology = grid.find('Topology')
    topo_type = (topology.get('TopologyType') or topology.get('Type') or '')
    if topo_type.upper() not in ('3DCORECTMESH', 'CORECTMESH'):
        raise ValueError(
            f"Unsupported TopologyType {topo_type!r} in {path!r}; only uniform "
            "structured grids (3DCORECTMesh) are supported. For other "
            "topologies, read the data yourself and construct a GriddedField."
        )
    dims_zyx = [int(v) for v in topology.get('NumberOfElements').split()]

    geometry = grid.find('Geometry')
    geo_type = (geometry.get('GeometryType') or geometry.get('Type') or '')
    if geo_type.upper() != 'ORIGIN_DXDYDZ':
        raise ValueError(
            f"Unsupported GeometryType {geo_type!r} in {path!r}; "
            "only ORIGIN_DXDYDZ is supported."
        )
    origin_zyx = spacing_zyx = None
    for item in geometry.findall('DataItem'):
        values = [float(v) for v in item.text.split()]
        if item.get('Name', '').lower() == 'origin':
            origin_zyx = values
        elif item.get('Name', '').lower() in ('spacing', 'dxdydz'):
            spacing_zyx = values
    if origin_zyx is None or spacing_zyx is None:
        raise ValueError(f"Origin/Spacing DataItems not found in {path!r}.")

    refs = {}
    for attr in grid.findall('Attribute'):
        name = attr.get('Name')
        if name not in components:
            continue
        item = attr.find('DataItem')
        if (item.get('Format') or '').upper() != 'HDF':
            raise ValueError(
                f"Attribute {name!r} in {path!r} is not HDF-backed "
                f"(Format={item.get('Format')!r})."
            )
        h5_file, _, h5_dset = item.text.strip().partition(':')
        refs[name] = (h5_file, h5_dset.lstrip('/'))
    missing = [c for c in components if c not in refs]
    if missing:
        raise KeyError(f"Attributes {missing} not found in {path!r}.")

    return (tuple(dims_zyx[::-1]), tuple(origin_zyx[::-1]),
            tuple(spacing_zyx[::-1]), refs)


def load_xdmf(path, components=('BX', 'BY', 'BZ'), h5_file=None, metadata=None):
    """
    Load a gridded magnetic field from an XDMF file and its HDF5 heavy data.

    Parameters
    ----------
    path : str
        XDMF (.xmf) file path describing a uniform structured grid.
    components : tuple of str, optional
        Attribute names of the (bx, by, bz) components. Default
        ('BX', 'BY', 'BZ').
    h5_file : str, optional
        Override for the HDF5 file path. By default the file referenced by
        the XDMF is resolved relative to the XDMF file's directory; pass this
        when the heavy data lives elsewhere or was renamed.
    metadata : dict, optional
        Extra provenance merged into ``GriddedField.metadata``.

    Returns
    -------
    GriddedField
    """
    h5py = _require_h5py()
    shape, origin, spacing, refs = _parse_xdmf_uniform_grid(path, components)

    comps = []
    xdmf_dir = os.path.dirname(os.path.abspath(path))
    for name in components:
        ref_file, dset = refs[name]
        target = h5_file if h5_file is not None else os.path.join(xdmf_dir, ref_file)
        with h5py.File(target, 'r') as f:
            arr = f[dset][()].transpose(2, 1, 0)
        if arr.shape != shape:
            raise ValueError(
                f"Dataset {dset!r} has grid shape {arr.shape}, but the XDMF "
                f"topology declares {shape}."
            )
        comps.append(np.ascontiguousarray(arr, dtype=arr.dtype.newbyteorder('=')))

    x, y, z = _axes_from_origin_spacing(origin, spacing, shape)
    meta = {'source': os.path.abspath(path), 'components': tuple(components)}
    if metadata:
        meta.update(metadata)
    return GriddedField(x, y, z, *comps, metadata=meta)
