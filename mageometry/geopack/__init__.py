# mageometry/geopack/__init__.py
"""
Vectorized geopack: magnetic field models and coordinate transforms.

This subpackage provides the field engine used by Mageometry: scalar and
NumPy-vectorized implementations of the Tsyganenko magnetospheric field
models (T89, T96, T01, T04), IGRF, dipole, coordinate transforms, and
field line tracing. It re-exports all public functions defined in
`geopack.py` so that they can be imported directly from
`mageometry.geopack`.
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

# Import the vectorized subpackage (enables: from mageometry.geopack import vectorized)
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

    # vectorized trace function
    "trace_vectorized",
]
