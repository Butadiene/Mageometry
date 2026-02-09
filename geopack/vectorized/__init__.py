# geopack/vectorized/__init__.py
"""
geopack.vectorized package initializer.

This module re-exports public vectorized functions and modules so that they can be
imported directly from `geopack.vectorized`.
"""

# vectorized modules
from . import (
    models,
    coordinates,
    coordinates_complex,
    igrf,
    field_line_geometry_vectorized,
    field_line_directional_derivatives,
)

# Re-export external field model functions (vectorized)
from .models import t89, t96, t01, t04

# Re-export vectorized IGRF functions
from .igrf import (
    igrf_geo_vectorized,
    igrf_gsm_vectorized,
    igrf_gsw_vectorized,
)

# Re-export vectorized coordinate transforms
from .coordinates import (
    gsmgse_vectorized,
    geigeo_vectorized,
    magsm_vectorized,
    smgsm_vectorized,
    geomag_vectorized,
    geogsm_vectorized,
    gswgsm_vectorized,
)

# Re-export vectorized "complex" coordinate transforms
from .coordinates_complex import (
    sphcar_vectorized,
    bspcar_vectorized,
    bcarsp_vectorized,
)

# Re-export extra exact implementations (vectorized utilities)
from .condip1_exact import condip1_exact

# Re-export field line geometry functions
from .field_line_geometry_vectorized import (
    field_line_tangent_vectorized,
    field_line_curvature_vectorized,
    field_line_normal_vectorized,
    field_line_binormal_vectorized,
    field_line_torsion_vectorized,
    field_line_frenet_frame_vectorized,
    field_line_geometry_complete_vectorized,
)

# Re-export directional derivatives utilities
from .field_line_directional_derivatives import (
    field_line_directional_derivatives_vectorized,
    verify_antisymmetry_relations,
    get_curvature_torsion_from_derivatives,
    verify_unit_vectors,
)

# core 側（同じグローバル状態を使う想定なら残してOK）
from ..geopack import (
    recalc,
    trace,
    dip,
    dip_gsw,
    igrf_gsm,
    igrf_gsw,
)

__all__ = [
    # vectorized modules
    "models",
    "coordinates",
    "coordinates_complex",
    "igrf",
    "field_line_geometry_vectorized",
    "field_line_directional_derivatives",

    # vectorized models
    "t89",
    "t96",
    "t01",
    "t04",

    # vectorized IGRF
    "igrf_geo_vectorized",
    "igrf_gsm_vectorized",
    "igrf_gsw_vectorized",

    # vectorized coordinate transforms
    "gsmgse_vectorized",
    "geigeo_vectorized",
    "magsm_vectorized",
    "smgsm_vectorized",
    "geomag_vectorized",
    "geogsm_vectorized",
    "gswgsm_vectorized",

    # vectorized "complex" coordinate transforms
    "sphcar_vectorized",
    "bspcar_vectorized",
    "bcarsp_vectorized",

    # vectorized utilities
    "condip1_exact",

    # field line geometry
    "field_line_tangent_vectorized",
    "field_line_curvature_vectorized",
    "field_line_normal_vectorized",
    "field_line_binormal_vectorized",
    "field_line_torsion_vectorized",
    "field_line_frenet_frame_vectorized",
    "field_line_geometry_complete_vectorized",

    # field line directional derivatives + helpers
    "field_line_directional_derivatives_vectorized",
    "verify_antisymmetry_relations",
    "get_curvature_torsion_from_derivatives",
    "verify_unit_vectors",

    # core functions re-exported
    "recalc",
    "trace",
    "dip",
    "dip_gsw",
    "igrf_gsm",
    "igrf_gsw",
]
