"""
Vectorized magnetospheric field models for efficient array processing.
"""

from .t89_vectorized import t89_vectorized
from .t96_vectorized import t96_vectorized
from .t01_vectorized import t01_vectorized
from .t04_vectorized import t04_vectorized
from .condip1_exact_vectorized import condip1_exact_vectorized

__all__ = ['t89_vectorized', 't96_vectorized', 't01_vectorized', 't04_vectorized', 'condip1_exact_vectorized']