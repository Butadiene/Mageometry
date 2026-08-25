# mageometry/io/xdmf.py
"""
Readers for XDMF-described and plain-HDF5 gridded magnetic field data.

XDMF (https://xdmf.org) is an XML metadata format used by many simulation
codes (and readable by ParaView/VisIt) that describes grid geometry and points
to "heavy data" stored in HDF5. `load_xdmf` supports the common case of a
uniform structured grid (``3DCORECTMesh`` topology with ``ORIGIN_DXDYDZ``
geometry) with node- or cell-centered scalar attributes for the field
components. Time series are handled by `load_xdmf_series`, which understands
both XDMF temporal collections (one file, several ``<Grid>`` elements with
``<Time>``) and ParaView ``.xmf.series`` JSON index files (one XDMF per
step); steps are loaded lazily, one at a time.

`load_hdf5` reads the HDF5 datasets directly when no XDMF file is available;
the grid geometry is then supplied by the caller.

All readers accept ``region`` (a bounding box in grid coordinates) and
``stride`` (subsampling factor) to read only part of a large dataset; the
selection is applied as an HDF5 hyperslab, so the full array never enters
memory. All readers return a `GriddedField`. HDF5 access requires the
optional `h5py` dependency (``pip install mageometry[io]``).

Axis-order convention: XDMF lists dimensions and Origin/DxDyDz values slowest
axis first (Z, Y, X for 3D), matching the C-order HDF5 dataset layout
``(nz, ny, nx)``. The readers transpose to the Mageometry convention
``(nx, ny, nz)``; use ``zyx_order=False`` in `load_hdf5` if your data is
already (nx, ny, nz).
"""

import json
import os
import xml.etree.ElementTree as ET

import numpy as np

from .gridded_field import GriddedField, FieldSeries, region_slices

__all__ = ["load_xdmf", "load_hdf5", "load_xdmf_series", "XdmfSeries"]


def _require_h5py():
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "Reading HDF5 data requires the optional dependency h5py. "
            "Install it with: pip install h5py"
        ) from None
    return h5py


def _uniform_axes(origin, spacing, shape, cell_centered=False):
    """Axes of a uniform grid; cell-centered data sits at (i + 1/2) * spacing."""
    half = 0.5 if cell_centered else 0.0
    return tuple(
        o + s * (np.arange(n, dtype=np.float64) + half)
        for o, s, n in zip(origin, spacing, shape)
    )


def _read_hyperslab(dset, slices_xyz, zyx_order=True):
    """Read the selected (x, y, z) slices of an HDF5 dataset as an (nx, ny, nz) array."""
    sx, sy, sz = slices_xyz
    if zyx_order:
        return np.asarray(dset[sz, sy, sx]).transpose(2, 1, 0)
    return np.asarray(dset[sx, sy, sz])


def _stride_tuple(stride):
    if np.isscalar(stride):
        stride = (stride, stride, stride)
    stride = tuple(int(s) for s in stride)
    if len(stride) != 3 or any(s < 1 for s in stride):
        raise ValueError("stride must be a positive int or a tuple of three positive ints.")
    return stride


def load_hdf5(path, datasets=('BX', 'BY', 'BZ'), origin=(0.0, 0.0, 0.0),
              spacing=(1.0, 1.0, 1.0), zyx_order=True, metadata=None,
              region=None, stride=1):
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
    region : sequence, optional
        ``((xmin, xmax), (ymin, ymax), (zmin, zmax))`` in grid coordinates;
        only nodes inside (inclusive) are read. ``None`` for an axis keeps
        its full extent.
    stride : int or tuple of int, optional
        Keep every ``stride``-th node per axis (coarsening). Default 1.

    Returns
    -------
    GriddedField
    """
    h5py = _require_h5py()
    stride = _stride_tuple(stride)
    comps = []
    with h5py.File(path, 'r') as f:
        for name in datasets:
            if name not in f:
                raise KeyError(
                    f"Dataset {name!r} not found in {path!r}; "
                    f"available: {sorted(f.keys())}"
                )
        dset0 = f[datasets[0]]
        shape = dset0.shape[::-1] if zyx_order else dset0.shape
        full_axes = _uniform_axes(origin, spacing, shape)
        slices = region_slices(full_axes, region, stride)
        for name in datasets:
            comps.append(_read_hyperslab(f[name], slices, zyx_order))

    axes = tuple(ax[sl] for ax, sl in zip(full_axes, slices))
    meta = {'source': os.path.abspath(path), 'datasets': tuple(datasets)}
    if metadata:
        meta.update(metadata)
    return GriddedField(*axes, *comps, metadata=meta)


# ---------------------------------------------------------------------------
# XDMF parsing
# ---------------------------------------------------------------------------

def _is_temporal_collection(grid):
    return ((grid.get('GridType') or '').lower() == 'collection'
            and (grid.get('CollectionType') or 'spatial').lower() == 'temporal')


def _grid_time(grid):
    t = grid.find('Time')
    if t is None or t.get('Value') is None:
        return None
    return float(t.get('Value'))


def _parse_uniform_grid(grid, components, where):
    """Parse a uniform-grid <Grid> element.

    Returns (shape_xyz, origin_xyz, spacing_xyz, cell_centered, refs) where
    refs maps component name -> (h5_filename, h5_dataset).
    """
    topology = grid.find('Topology')
    if topology is None:
        raise ValueError(f"No <Topology> in grid {where}.")
    topo_type = (topology.get('TopologyType') or topology.get('Type') or '')
    if topo_type.upper() not in ('3DCORECTMESH', 'CORECTMESH'):
        raise ValueError(
            f"Unsupported TopologyType {topo_type!r} in {where}; only uniform "
            "structured grids (3DCORECTMesh) are supported. For other "
            "topologies, read the data yourself and construct a GriddedField."
        )
    dims_zyx = [int(v) for v in topology.get('NumberOfElements').split()]

    geometry = grid.find('Geometry')
    if geometry is None:
        raise ValueError(f"No <Geometry> in grid {where}.")
    geo_type = (geometry.get('GeometryType') or geometry.get('Type') or '')
    if geo_type.upper() != 'ORIGIN_DXDYDZ':
        raise ValueError(
            f"Unsupported GeometryType {geo_type!r} in {where}; "
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
        raise ValueError(f"Origin/Spacing DataItems not found in {where}.")

    refs = {}
    centers = {}
    for attr in grid.findall('Attribute'):
        name = attr.get('Name')
        if name not in components:
            continue
        item = attr.find('DataItem')
        if (item.get('Format') or '').upper() != 'HDF':
            raise ValueError(
                f"Attribute {name!r} in {where} is not HDF-backed "
                f"(Format={item.get('Format')!r})."
            )
        h5_file, _, h5_dset = item.text.strip().partition(':')
        refs[name] = (h5_file, h5_dset.lstrip('/'))
        centers[name] = (attr.get('Center') or 'Node').lower()
    missing = [c for c in components if c not in refs]
    if missing:
        raise KeyError(f"Attributes {missing} not found in {where}.")

    center_set = set(centers.values())
    if len(center_set) != 1 or not center_set <= {'node', 'cell'}:
        raise ValueError(
            f"Components must share one centering of 'Node' or 'Cell'; "
            f"got {centers} in {where}."
        )
    cell_centered = center_set == {'cell'}

    shape = tuple(dims_zyx[::-1])
    if cell_centered:
        shape = tuple(n - 1 for n in shape)
    return (shape, tuple(origin_zyx[::-1]), tuple(spacing_zyx[::-1]),
            cell_centered, refs)


def _load_grid(grid, components, xdmf_dir, where, h5_file=None,
               metadata=None, region=None, stride=1):
    h5py = _require_h5py()
    stride = _stride_tuple(stride)
    shape, origin, spacing, cell_centered, refs = _parse_uniform_grid(
        grid, components, where)
    full_axes = _uniform_axes(origin, spacing, shape, cell_centered)
    slices = region_slices(full_axes, region, stride)

    comps = []
    for name in components:
        ref_file, dset = refs[name]
        target = h5_file if h5_file is not None else os.path.join(xdmf_dir, ref_file)
        with h5py.File(target, 'r') as f:
            stored = f[dset].shape[::-1]
            if stored != shape:
                raise ValueError(
                    f"Dataset {dset!r} has grid shape {stored}, but the XDMF "
                    f"topology declares {shape} for {'cell' if cell_centered else 'node'}-"
                    "centered data."
                )
            comps.append(_read_hyperslab(f[dset], slices))

    axes = tuple(ax[sl] for ax, sl in zip(full_axes, slices))
    meta = {'source': where, 'components': tuple(components),
            'center': 'cell' if cell_centered else 'node'}
    t = _grid_time(grid)
    if t is not None:
        meta['time'] = t
    if metadata:
        meta.update(metadata)
    return GriddedField(*axes, *comps, metadata=meta)


def _root_grid(path):
    root = ET.parse(path).getroot()
    grid = root.find('.//Grid')
    if grid is None:
        raise ValueError(f"No <Grid> element found in {path!r}.")
    return grid


def load_xdmf(path, components=('BX', 'BY', 'BZ'), h5_file=None, metadata=None,
              region=None, stride=1):
    """
    Load a gridded magnetic field from an XDMF file and its HDF5 heavy data.

    Parameters
    ----------
    path : str
        XDMF (.xmf) file path describing a uniform structured grid.
    components : tuple of str, optional
        Attribute names of the (bx, by, bz) components. Default
        ('BX', 'BY', 'BZ'). Node- and cell-centered attributes are both
        accepted (cell-centered values are placed at the cell centers).
    h5_file : str, optional
        Override for the HDF5 file path. By default the file referenced by
        the XDMF is resolved relative to the XDMF file's directory; pass this
        when the heavy data lives elsewhere or was renamed.
    metadata : dict, optional
        Extra provenance merged into ``GriddedField.metadata``.
    region : sequence, optional
        ``((xmin, xmax), (ymin, ymax), (zmin, zmax))`` in grid coordinates;
        only nodes inside (inclusive) are read from the HDF5 file. ``None``
        for an axis keeps its full extent.
    stride : int or tuple of int, optional
        Keep every ``stride``-th node per axis (coarsening). Default 1.

    Returns
    -------
    GriddedField
        ``metadata['time']`` is set when the grid carries a ``<Time>``.

    Raises
    ------
    ValueError
        For unsupported topologies/geometries, and for time-series files
        (use `load_xdmf_series`).
    """
    grid = _root_grid(path)
    if _is_temporal_collection(grid):
        raise ValueError(
            f"{path!r} is an XDMF temporal collection (time series); "
            "use load_xdmf_series() and pick a step."
        )
    xdmf_dir = os.path.dirname(os.path.abspath(path))
    return _load_grid(grid, components, xdmf_dir, os.path.abspath(path),
                      h5_file=h5_file, metadata=metadata, region=region,
                      stride=stride)


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------

class XdmfSeries(FieldSeries):
    """
    A lazily loaded time series of XDMF-described gridded fields.

    Created by `load_xdmf_series`; see `FieldSeries` for the interface
    (``times``, ``series[i]``, ``series.at(t)``, iteration, slicing).
    Reader options given to `load_xdmf_series` apply to every step.
    """

    def __getitem__(self, i):
        if isinstance(i, slice):
            return XdmfSeries(self._steps[i])
        return super().__getitem__(i)


def load_xdmf_series(path, components=('BX', 'BY', 'BZ'), metadata=None,
                     region=None, stride=1):
    """
    Open a time series of XDMF-described gridded fields (lazily).

    Two layouts are recognized:

    - a ParaView ``.xmf.series`` JSON index
      (``{"file-series-version": "1.0", "files": [{"name": ..., "time": ...}, ...]}``),
      each entry naming a single-grid XDMF file relative to the index;
    - one XDMF file whose ``<Grid GridType="Collection"
      CollectionType="Temporal">`` contains one uniform ``<Grid>`` per step,
      each with a ``<Time Value="..."/>``.

    Parameters
    ----------
    path : str
        ``.xmf.series`` index or temporal-collection ``.xmf`` file.
    components, metadata, region, stride
        Passed to the per-step reader; see `load_xdmf`.

    Returns
    -------
    XdmfSeries
        Index it (``series[i]``), iterate over it, or use ``series.at(time)``
        to obtain a `GriddedField` per step.
    """
    load_kwargs = dict(components=components, metadata=metadata,
                       region=region, stride=stride)
    base_dir = os.path.dirname(os.path.abspath(path))
    steps = []

    if path.endswith('.series') or _looks_like_json(path):
        with open(path) as f:
            index = json.load(f)
        entries = index.get('files')
        if not isinstance(entries, list):
            raise ValueError(f"{path!r} is not a file-series index (no 'files' list).")
        for entry in entries:
            name = entry['name']
            step_path = name if os.path.isabs(name) else os.path.join(base_dir, name)
            t = entry.get('time')
            t = float(t) if t is not None else None

            def loader(step_path=step_path, t=t):
                # The index carries the time; the per-step file usually does not.
                kw = dict(load_kwargs)
                meta = dict(kw.get('metadata') or {})
                if t is not None:
                    meta.setdefault('time', t)
                kw['metadata'] = meta
                return load_xdmf(step_path, **kw)

            steps.append((t, step_path, loader))
        return XdmfSeries(steps)

    grid = _root_grid(path)
    if not _is_temporal_collection(grid):
        raise ValueError(
            f"{path!r} is neither a .xmf.series index nor an XDMF temporal "
            "collection; for a single grid use load_xdmf()."
        )
    abs_path = os.path.abspath(path)
    for k, child in enumerate(grid.findall('Grid')):
        where = f"{abs_path}[grid {k}]"

        def loader(child=child, where=where):
            return _load_grid(child, xdmf_dir=base_dir, where=where, **load_kwargs)

        steps.append((_grid_time(child), where, loader))
    if not steps:
        raise ValueError(f"Temporal collection in {path!r} contains no grids.")
    return XdmfSeries(steps)


def _looks_like_json(path):
    with open(path, 'rb') as f:
        head = f.read(64).lstrip()
    return head.startswith(b'{')
