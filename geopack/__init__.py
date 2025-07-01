"""
Geopack - Python implementation of Tsyganenko magnetospheric field models.

This package provides both scalar and vectorized implementations of various
magnetospheric field models including T89, T96, T01, and T04.
"""

# Core functionality
from .geopack import (
    recalc, igrf_gsw, igrf_gsm, igrf_geo, dip, sun,
    sphcar, bcarsp, bspcar, 
    geomag, magsm, gswgsm, gsmgse, smgsm,
    geigeo, geogsm, geodgeo
)

# Scalar models
from .models import t89, t96, t01, t04

# Vectorized models
from .vectorized import (
    t89_vectorized, t96_vectorized, 
    t01_vectorized, t04_vectorized,
    condip1_exact_vectorized
)

# Vectorized coordinate transformations
from .coordinates_vectorized import (
    gsmgse_vectorized, geigeo_vectorized, magsm_vectorized,
    smgsm_vectorized, geomag_vectorized, geogsm_vectorized,
    gswgsm_vectorized
)

from .coordinates_vectorized_complex import (
    sphcar_vectorized, bspcar_vectorized, bcarsp_vectorized
)

# Vectorized IGRF functions
from .igrf_vectorized import (
    igrf_geo_vectorized, igrf_gsm_vectorized, igrf_gsw_vectorized
)


__version__ = '1.0.13'

__all__ = [
    # Core functions
    'recalc', 'igrf_gsw', 'igrf_gsm', 'igrf_geo', 'dip', 'sun',
    'sphcar', 'bcarsp', 'bspcar',
    'geomag', 'magsm', 'gswgsm', 'gsmgse', 'smgsm',
    'geigeo', 'geogsm', 'geodgeo',
    # Scalar models
    't89', 't96', 't01', 't04',
    # Vectorized models
    't89_vectorized', 't96_vectorized', 
    't01_vectorized', 't04_vectorized',
    'condip1_exact_vectorized',
    # Vectorized coordinate transformations
    'gsmgse_vectorized', 'geigeo_vectorized', 'magsm_vectorized',
    'smgsm_vectorized', 'geomag_vectorized', 'geogsm_vectorized',
    'gswgsm_vectorized', 'sphcar_vectorized', 'bspcar_vectorized', 
    'bcarsp_vectorized',
    # Vectorized IGRF functions
    'igrf_geo_vectorized', 'igrf_gsm_vectorized', 'igrf_gsw_vectorized'
]