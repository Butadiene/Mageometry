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
- `load_vtk` — VTK ImageData / RectilinearGrid files (``.vti``, ``.vtr``);
  needs the optional `pyvista` dependency

All readers take ``region`` / ``stride`` to read a sub-box or coarsened
grid (as an HDF5 hyperslab where the format allows it);
`GriddedField.subvolume` does the same in memory. The XDMF/HDF5 readers
need the optional `h5py` dependency.

For data in any other format, write a small ``load_<format>(path) ->
GriddedField`` following ``docs/simulation_data_formats.md``; the building
blocks here are `read_fortran_records` (Fortran unformatted files),
`FieldSeries.from_files` (lazy time series from per-step files and your
loader), and `GriddedField.divergence` (a sanity check that catches
permuted axes or components). Additional formats (CDF, raw binaries, ...)
should follow the same pattern: parse, build arrays, return a
`GriddedField`.
"""

from .gridded_field import GriddedField, FieldSeries, region_slices
from .binary import iter_fortran_records, read_fortran_records
from .xdmf import load_xdmf, load_hdf5, load_xdmf_series, XdmfSeries
from .vtk import load_vtk

__all__ = [
    "GriddedField",
    "FieldSeries",
    "region_slices",
    "iter_fortran_records",
    "read_fortran_records",
    "load_xdmf",
    "load_hdf5",
    "load_xdmf_series",
    "load_vtk",
    "XdmfSeries",
]
