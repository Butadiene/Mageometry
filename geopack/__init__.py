"""
Geopack - Python implementation of Tsyganenko magnetospheric field models.

This package provides both scalar and vectorized implementations of various
magnetospheric field models including T89, T96, T01, and T04.
"""

# Core functionality
from .geopack import (
    recalc, igrf_gsw, igrf_geo, dip, sun,
    sphcar, bcarsp, bspcar, 
    geomag, magsm, smgsw, geogsw, gswgse, geigse,
    geigeo, geigsw, geosm, gsegsw, geodgeo
)

# Scalar models
from .models import t89, t96, t01, t04

# Vectorized models
from .vectorized import (
    t89_vectorized, t96_vectorized, 
    t01_vectorized, t04_vectorized,
    condip1_exact_vectorized
)

__version__ = '1.0.12'

__all__ = [
    # Core functions
    'recalc', 'igrf_gsw', 'igrf_geo', 'dip', 'sun',
    'sphcar', 'bcarsp', 'bspcar',
    'geomag', 'magsm', 'smgsw', 'geogsw', 'gswgse', 'geigse',
    'geigeo', 'geigsw', 'geosm', 'gsegsw', 'geodgeo',
    # Scalar models
    't89', 't96', 't01', 't04',
    # Vectorized models
    't89_vectorized', 't96_vectorized', 
    't01_vectorized', 't04_vectorized',
    'condip1_exact_vectorized'
]