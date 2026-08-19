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

Planned subpackages: `mageometry.io` (simulation output readers) and
`mageometry.viz` (visualization).
"""

__version__ = '0.1.0.dev0'

# Field sources / engines
from . import geopack
from . import fields
from .fields import geopack_field

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

    # field source adapters
    "geopack_field",

    # Frenet-Serret frame
    "field_line_tangent",
    "field_line_curvature",
    "field_line_normal",
    "field_line_binormal",
    "field_line_torsion",
    "field_line_frenet_frame",
    "field_line_geometry_complete",

    # directional derivatives and verification utilities
    "field_line_directional_derivatives",
    "verify_antisymmetry_relations",
    "get_curvature_torsion_from_derivatives",
    "verify_unit_vectors",
]
