# Vectorized Model Interface Alignment

## Overview

The T01 and T96 vectorized implementations have been aligned to use identical interfaces for consistency and ease of use.

## Common Interface Pattern

Both `t01_vectorized` and `t96_vectorized` follow the same pattern:

### Function Signature
```python
def model_vectorized(parmod, ps, x, y, z):
    """
    Parameters
    ----------
    parmod : array_like
        10-element array containing model parameters
    ps : float
        Geodipole tilt angle in radians
    x, y, z : array_like
        GSM coordinates in Re (Earth radii)
        
    Returns
    -------
    bx, by, bz : ndarray
        Magnetic field components in GSM system (nT)
    """
```

### Key Features

1. **Scalar Input Handling**
   ```python
   # Track if all inputs were scalar
   scalar_input = np.isscalar(x) and np.isscalar(y) and np.isscalar(z)
   
   # Convert inputs to numpy arrays
   x = np.atleast_1d(x)
   y = np.atleast_1d(y)
   z = np.atleast_1d(z)
   ```

2. **Scalar Output Handling**
   ```python
   # Return scalar if input was scalar
   if scalar_input:
       return bx.item(), by.item(), bz.item()
   else:
       return bx, by, bz
   ```

3. **Mixed Input Support**
   - Handles mixed scalar/array inputs (e.g., array x, scalar y, array z)
   - Proper broadcasting throughout the implementation

4. **Safe Division**
   - Uses `np.where` instead of `np.divide` with `out` parameter
   - Ensures proper broadcasting for all input combinations

## Parameter Arrays

### T96 Parameters (parmod)
- `[0]` - Solar wind pressure pdyn (nPa)
- `[1]` - Dst (nT)
- `[2]` - IMF By (nT)
- `[3]` - IMF Bz (nT)
- `[4-9]` - Unused

### T01 Parameters (parmod)
- `[0]` - Solar wind pressure pdyn (nPa)
- `[1]` - Dst (nT)
- `[2]` - IMF By (nT)
- `[3]` - IMF Bz (nT)
- `[4]` - G1 index
- `[5]` - G2 index
- `[6-9]` - Unused

## Usage Examples

```python
import numpy as np
from geopack.t01_vectorized import t01_vectorized
from geopack.t96_vectorized import t96_vectorized

# Scalar inputs - returns scalars
bx, by, bz = t01_vectorized(parmod, ps, 5.0, 2.0, 1.0)
# type(bx) == float

# Array inputs - returns arrays
x = np.array([5.0, 6.0, 7.0])
y = np.array([2.0, 3.0, 4.0])
z = np.array([1.0, 1.5, 2.0])
bx, by, bz = t96_vectorized(parmod, ps, x, y, z)
# type(bx) == numpy.ndarray, shape == (3,)

# Mixed inputs - returns arrays
bx, by, bz = t01_vectorized(parmod, ps, x, 2.0, z)
# Broadcasting handles scalar y with array x and z
```

## Benefits

1. **Consistency**: Same interface pattern across all vectorized models
2. **Flexibility**: Handles scalars, arrays, and mixed inputs seamlessly
3. **Performance**: Optimized for array operations while maintaining scalar compatibility
4. **Drop-in Replacement**: Can replace scalar versions with minimal code changes

## Implementation Principles

Both models follow the same vectorization principles:
1. All functions accept NumPy arrays for x, y, z coordinates
2. Conditional logic uses `np.where` instead of if/else
3. Safe division using `np.where` for proper broadcasting
4. No global variables - all parameters passed explicitly
5. Proper array initialization with `np.zeros_like()`

This alignment ensures that users can switch between models easily and that future vectorized implementations follow the same pattern.