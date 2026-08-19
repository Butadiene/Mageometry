# mageometry/io/__init__.py
"""
Magnetic field input from simulation output files.

The extensibility contract is `GriddedField`: any file-format reader only has
to produce three coordinate axes plus the field components on that grid and
construct a `GriddedField`. Its ``.field()`` method then yields the
``field(x, y, z) -> (bx, by, bz)`` callable consumed by all
`mageometry.geometry` functions.

Currently provided readers:

- `load_xdmf` — XDMF-described uniform grids with HDF5 heavy data
- `load_hdf5` — plain HDF5 datasets with caller-supplied grid geometry

Both need the optional `h5py` dependency. Additional formats (VTK, CDF,
raw binaries, ...) should follow the same pattern: parse, build arrays,
return a `GriddedField`.
"""

from .gridded_field import GriddedField
from .xdmf import load_xdmf, load_hdf5

__all__ = [
    "GriddedField",
    "load_xdmf",
    "load_hdf5",
]
