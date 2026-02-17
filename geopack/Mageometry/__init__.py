# geopack/Mageometry/__init__.py
"""
Magnetic field line geometry analysis (Mageometry).

Provides Frenet-Serret frame calculations (tangent, normal, binormal,
curvature, torsion) and directional derivative analysis for magnetic
field lines.  All functions accept vectorized (NumPy array) inputs.
"""

from . import (
    field_line_geometry,
    field_line_directional_derivatives,
)

# Frenet-Serret frame: tangent, curvature, normal, binormal, torsion
from .field_line_geometry import (
    field_line_tangent,
    field_line_curvature,
    field_line_normal,
    field_line_binormal,
    field_line_torsion,
    field_line_frenet_frame,
    field_line_geometry_complete,
)

# Directional derivatives of the Frenet-Serret frame and verification utilities
from .field_line_directional_derivatives import (
    field_line_directional_derivatives,
    verify_antisymmetry_relations,
    get_curvature_torsion_from_derivatives,
    verify_unit_vectors,
)

__all__ = [
    "field_line_geometry",
    "field_line_directional_derivatives",

    # field line geometry
    "field_line_tangent",
    "field_line_curvature",
    "field_line_normal",
    "field_line_binormal",
    "field_line_torsion",
    "field_line_frenet_frame",
    "field_line_geometry_complete",

    # field line directional derivatives + helpers
    "field_line_directional_derivatives",
    "verify_antisymmetry_relations",
    "get_curvature_torsion_from_derivatives",
    "verify_unit_vectors",
]
