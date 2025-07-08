"""
Vectorized magnetospheric field models for efficient array processing.
"""

from .t89_vectorized import t89_vectorized
from .t96_vectorized import t96_vectorized
from .t01_vectorized import t01_vectorized
from .t04_vectorized import t04_vectorized
from .condip1_exact_vectorized import condip1_exact_vectorized
from .field_line_geometry_vectorized import (
    field_line_tangent_vectorized,
    field_line_curvature_vectorized,
    field_line_normal_vectorized,
    field_line_binormal_vectorized,
    field_line_torsion_vectorized,
    field_line_frenet_frame_vectorized,
    field_line_geometry_complete_vectorized
)
from .field_line_directional_derivatives_new import (
    field_line_directional_derivatives_vectorized,
    verify_antisymmetry_relations,
    get_curvature_torsion_from_derivatives,
    verify_unit_vectors
)

__all__ = [
    't89_vectorized', 't96_vectorized', 't01_vectorized', 't04_vectorized', 
    'condip1_exact_vectorized',
    'field_line_tangent_vectorized', 'field_line_curvature_vectorized',
    'field_line_normal_vectorized', 'field_line_binormal_vectorized',
    'field_line_torsion_vectorized', 'field_line_frenet_frame_vectorized',
    'field_line_geometry_complete_vectorized',
    'field_line_directional_derivatives_vectorized',
    'verify_antisymmetry_relations',
    'get_curvature_torsion_from_derivatives',
    'verify_unit_vectors'
]