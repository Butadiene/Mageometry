# geopack/__init__.py
"""
geopack package initializer.

This module re-exports all public functions defined in geopack/geopack.py,
so that they can be imported directly from `geopack`.
"""

__version__ = '2.0.0'

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

# Import the vectorized subpackage (enables: from geopack import vectorized)
from . import vectorized

# Vectorized modules (aliased to avoid name collisions with scalar counterparts)
from .vectorized import (
    models as vectorized_models,
    coordinates as vectorized_coordinates,
    coordinates_complex as vectorized_coordinates_complex,
    igrf as vectorized_igrf,
)

# vectorized IGRF
from .vectorized.igrf import (
    igrf_geo as igrf_geo_vectorized,
    igrf_gsm as igrf_gsm_vectorized,
    igrf_gsw as igrf_gsw_vectorized,
)

# vectorized coordinate transforms
from .vectorized.coordinates import (
    gsmgse as gsmgse_vectorized,
    geigeo as geigeo_vectorized,
    magsm as magsm_vectorized,
    smgsm as smgsm_vectorized,
    geomag as geomag_vectorized,
    geogsm as geogsm_vectorized,
    gswgsm as gswgsm_vectorized,
)

# vectorized "complex" coordinate transforms
from .vectorized.coordinates_complex import (
    sphcar as sphcar_vectorized,
    bspcar as bspcar_vectorized,
    bcarsp as bcarsp_vectorized,
)

# vectorized utilities
from .vectorized.condip1_exact import condip1_exact

# Vectorized external models (aliased to avoid name collisions with scalar versions)
from .vectorized.models import (
    t89 as t89_vectorized,
    t96 as t96_vectorized,
    t01 as t01_vectorized,
    t04 as t04_vectorized,
)

# trace functions
from .vectorized.trace import trace as trace_vectorized


# -----------------------------
# Mageometry (re-export)
# Magnetic field line geometry: Frenet-Serret frame and directional derivatives
# -----------------------------

# Frenet-Serret frame: tangent, curvature, normal, binormal, torsion
from .Mageometry import field_line_geometry
from .Mageometry.field_line_geometry import (
    field_line_tangent,
    field_line_curvature,
    field_line_normal,
    field_line_binormal,
    field_line_torsion,
    field_line_frenet_frame,
    field_line_geometry_complete,
)

# Directional derivatives of the Frenet-Serret frame and verification utilities
from .Mageometry import field_line_directional_derivatives

from .Mageometry.field_line_directional_derivatives import (
    verify_antisymmetry_relations,
    get_curvature_torsion_from_derivatives,
    verify_unit_vectors,
)

# Backward-compatible aliases: top-level names with the _vectorized suffix,
# as published before the move to geopack.Mageometry
from .Mageometry.field_line_geometry import (
    field_line_tangent as field_line_tangent_vectorized,
    field_line_curvature as field_line_curvature_vectorized,
    field_line_normal as field_line_normal_vectorized,
    field_line_binormal as field_line_binormal_vectorized,
    field_line_torsion as field_line_torsion_vectorized,
    field_line_frenet_frame as field_line_frenet_frame_vectorized,
    field_line_geometry_complete as field_line_geometry_complete_vectorized,
)
from .Mageometry.field_line_directional_derivatives import (
    field_line_directional_derivatives as field_line_directional_derivatives_vectorized,
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

    # Mageometry: Frenet-Serret frame
    "field_line_geometry",
    "field_line_tangent",
    "field_line_curvature",
    "field_line_normal",
    "field_line_binormal",
    "field_line_torsion",
    "field_line_frenet_frame",
    "field_line_geometry_complete",

    # Mageometry: directional derivatives and verification utilities
    "field_line_directional_derivatives",
    "verify_antisymmetry_relations",
    "get_curvature_torsion_from_derivatives",
    "verify_unit_vectors",

    # Mageometry: backward-compatible _vectorized aliases
    "field_line_tangent_vectorized",
    "field_line_curvature_vectorized",
    "field_line_normal_vectorized",
    "field_line_binormal_vectorized",
    "field_line_torsion_vectorized",
    "field_line_frenet_frame_vectorized",
    "field_line_geometry_complete_vectorized",
    "field_line_directional_derivatives_vectorized",

    # vectorized trace function
    "trace_vectorized",
]
