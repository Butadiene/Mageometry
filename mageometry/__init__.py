# mageometry/__init__.py
"""
Mageometry: magnetic field line geometry analysis.

The primary API is the field line geometry toolkit (Frenet-Serret frames,
curvature, torsion, and directional derivatives), re-exported here from
`mageometry.geometry`.

The geometry functions take the magnetic field as a callable
``field(x, y, z) -> (bx, by, bz)``. Field sources:

- `mageometry.fields` — adapters building such callables (`geopack_field`)
- `mageometry.geopack` — vectorized geopack (Tsyganenko models T89/T96/T01/T04,
  IGRF, dipole, coordinate transforms, field line tracing)
- `mageometry.io` — gridded fields from simulation output files
  (`GriddedField`, `load_xdmf`, `load_hdf5`)

`mageometry.tracing` traces field lines through any such field callable
(`trace_field_lines`).

Planned subpackage: `mageometry.viz` (visualization).
"""

__version__ = '0.1.0.dev0'

# Field sources / engines
from . import geopack
from . import fields
from . import io
from .fields import geopack_field
from .io import GriddedField, load_xdmf, load_hdf5

# Field line tracing (generic, field-callable based)
from . import tracing
from .tracing import trace_field_lines, FieldLineTrace

# Geometry analysis (primary API)
from . import geometry
from .geometry import (
    # Frenet-Serret frame
    field_line_tangent,
    field_line_curvature,
    field_line_normal,
    field_line_binormal,
    field_line_torsion,
    field_line_frenet_frame,
    field_line_geometry_complete,
    field_line_frame_quality,
    # Directional derivatives and verification utilities
    field_line_directional_derivatives,
    verify_antisymmetry_relations,
    get_curvature_torsion_from_derivatives,
    verify_unit_vectors,
)

__all__ = [
    # subpackages
    "geometry",
    "geopack",
    "fields",
    "io",
    "tracing",

    # field source adapters
    "geopack_field",

    # simulation data input
    "GriddedField",
    "load_xdmf",
    "load_hdf5",

    # field line tracing
    "trace_field_lines",
    "FieldLineTrace",

    # Frenet-Serret frame
    "field_line_tangent",
    "field_line_curvature",
    "field_line_normal",
    "field_line_binormal",
    "field_line_torsion",
    "field_line_frenet_frame",
    "field_line_geometry_complete",
    "field_line_frame_quality",

    # directional derivatives and verification utilities
    "field_line_directional_derivatives",
    "verify_antisymmetry_relations",
    "get_curvature_torsion_from_derivatives",
    "verify_unit_vectors",
]
