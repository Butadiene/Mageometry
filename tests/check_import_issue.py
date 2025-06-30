"""
Check if there's an import or caching issue.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force reload
import importlib
import geopack.ring_current_vectorized
importlib.reload(geopack.ring_current_vectorized)

from geopack.ring_current_vectorized import full_rc_vectorized

# Check the source
import inspect
print("full_rc_vectorized source file:", inspect.getfile(full_rc_vectorized))
print("\nFirst 20 lines of full_rc_vectorized:")
print(inspect.getsource(full_rc_vectorized)[:1000])