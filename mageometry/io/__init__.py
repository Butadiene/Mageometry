# mageometry/io/__init__.py
"""
Magnetic field input from simulation output files.

The extensibility contract is `GriddedField`: any file-format reader only has
to produce three coordinate axes plus the field components on that grid and
construct a `GriddedField`. Its ``.field()`` method then yields the
``field(x, y, z) -> (bx, by, bz)`` callable consumed by all
`mageometry.geometry` functions.

Currently provided readers:

- `load_xdmf` — XDMF-described uniform grids (node- or cell-centered) with
  HDF5 heavy data
- `load_hdf5` — plain HDF5 datasets with caller-supplied grid geometry
- `load_xdmf_series` — time series (XDMF temporal collections or ParaView
  ``.xmf.series`` indexes), loaded lazily step by step as an `XdmfSeries`

All readers take ``region`` / ``stride`` to read a sub-box or coarsened grid
as an HDF5 hyperslab; `GriddedField.subvolume` does the same in memory. All
need the optional `h5py` dependency. Additional formats (VTK, CDF,
raw binaries, ...) should follow the same pattern: parse, build arrays,
return a `GriddedField`.
"""

from .gridded_field import GriddedField, region_slices
from .xdmf import load_xdmf, load_hdf5, load_xdmf_series, XdmfSeries

__all__ = [
    "GriddedField",
    "region_slices",
    "load_xdmf",
    "load_hdf5",
    "load_xdmf_series",
    "XdmfSeries",
]
