# geopack/__init__.py
"""
geopack package initializer.

This module re-exports all public functions defined in geopack/geopack.py,
so that they can be imported directly from `geopack`.
"""

from .geopack import (
    update_igrf,
    init_igrf,
    load_igrf,
    igrf_gsw,
    igrf_gsm,
    igrf_geo,
    dip,
    dip_gsw,
    recalc,
    sun,
    gswgsm,
    geomag,
    geigeo,
    magsm,
    gsmgse,
    smgsm,
    geogsm,
    geodgeo,
    sphcar,
    bspcar,
    bcarsp,
    call_external_model,
    call_internal_model,
    rhand,
    step,
    trace,
    shuetal_mgnp,
    t96_mgnp,
)

# Re-export external field model functions (imported/used by geopack.py)
from .models import t89, t96, t01, t04

# -----------------------------
# vectorized (re-export)
# -----------------------------

# サブパッケージそのもの（from geopack import vectorized を可能に）
from . import vectorized

# vectorized 側のモジュール（scalar 側と名前が衝突しやすいので別名で公開）
from .vectorized import (
    models as vectorized_models,
    coordinates as vectorized_coordinates,
    coordinates_complex as vectorized_coordinates_complex,
    igrf as vectorized_igrf,
)

# vectorized IGRF
from .vectorized.igrf import (
    igrf_geo_vectorized,
    igrf_gsm_vectorized,
    igrf_gsw_vectorized,
)

# vectorized coordinate transforms
from .vectorized.coordinates import (
    gsmgse_vectorized,
    geigeo_vectorized,
    magsm_vectorized,
    smgsm_vectorized,
    geomag_vectorized,
    geogsm_vectorized,
    gswgsm_vectorized,
)

# vectorized "complex" coordinate transforms
from .vectorized.coordinates_complex import (
    sphcar_vectorized,
    bspcar_vectorized,
    bcarsp_vectorized,
)

# vectorized utilities
from .vectorized.condip1_exact import condip1_exact

# vectorized field line geometry
from .vectorized.field_line_geometry_vectorized import (
    field_line_tangent_vectorized,
    field_line_curvature_vectorized,
    field_line_normal_vectorized,
    field_line_binormal_vectorized,
    field_line_torsion_vectorized,
    field_line_frenet_frame_vectorized,
    field_line_geometry_complete_vectorized,
)

# vectorized field line directional derivatives + helpers
from .vectorized.field_line_directional_derivatives import (
    field_line_directional_derivatives_vectorized,
    verify_antisymmetry_relations,
    get_curvature_torsion_from_derivatives,
    verify_unit_vectors,
)

# vectorized external models（scalar と同名衝突するので alias）
from .vectorized.models import (
    t89 as t89_vectorized,
    t96 as t96_vectorized,
    t01 as t01_vectorized,
    t04 as t04_vectorized,
)

__all__ = [
    # geopack.py functions
    "update_igrf",
    "init_igrf",
    "load_igrf",
    "igrf_gsw",
    "igrf_gsm",
    "igrf_geo",
    "dip",
    "dip_gsw",
    "recalc",
    "sun",
    "gswgsm",
    "geomag",
    "geigeo",
    "magsm",
    "gsmgse",
    "smgsm",
    "geogsm",
    "geodgeo",
    "sphcar",
    "bspcar",
    "bcarsp",
    "call_external_model",
    "call_internal_model",
    "rhand",
    "step",
    "trace",
    "shuetal_mgnp",
    "t96_mgnp",

    # scalar models
    "t89",
    "t96",
    "t01",
    "t04",

    # vectorized package + modules (namespaced)
    "vectorized",
    "vectorized_models",
    "vectorized_coordinates",
    "vectorized_coordinates_complex",
    "vectorized_igrf",

    # vectorized IGRF
    "igrf_geo_vectorized",
    "igrf_gsm_vectorized",
    "igrf_gsw_vectorized",

    # vectorized models
    "t89_vectorized",
    "t96_vectorized",
    "t01_vectorized",
    "t04_vectorized",

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

    # vectorized field line geometry
    "field_line_tangent_vectorized",
    "field_line_curvature_vectorized",
    "field_line_normal_vectorized",
    "field_line_binormal_vectorized",
    "field_line_torsion_vectorized",
    "field_line_frenet_frame_vectorized",
    "field_line_geometry_complete_vectorized",

    # vectorized field line directional derivatives + helpers
    "field_line_directional_derivatives_vectorized",
    "verify_antisymmetry_relations",
    "get_curvature_torsion_from_derivatives",
    "verify_unit_vectors",
]
