# T01 Vectorization Issues Report

## Summary

The T01 vectorized implementation provided in `direction_vectorize_6.md` has significant issues that prevent it from producing accurate results compared to the scalar version.

## Issues Found

### 1. Broadcasting Errors

The vectorized code has multiple broadcasting errors when handling array inputs:

- **Issue**: Using `np.eye(3)[0]` as fallback values in `np.where()` statements
- **Location**: `deformed()` function around line 583
- **Fix Applied**: Changed to `np.ones_like()` and `np.zeros_like()` for proper broadcasting

### 2. Indexing Errors

The code attempts to index 1D arrays as if they were 2D:

- **Issue**: `dx1del[1]`, `dy1del[0]`, etc. when these are 1D arrays
- **Location**: `deformed()` function around line 600-606
- **Error**: `IndexError: index 1 is out of bounds for axis 0 with size 1`

### 3. Significant Accuracy Discrepancies

When comparing single point calculations:

```
Position: (5.0, 2.0, 1.0)
Scalar result: Bx=1.310518, By=-0.406567, Bz=-2.128649
Vector result: Bx=0.000000, By=0.000000, Bz=-8.626954

Absolute differences:
ΔBx = 1.31e+00
ΔBy = 4.07e-01
ΔBz = 6.50e+00
```

The differences are orders of magnitude larger than acceptable tolerance (1e-6 nT).

### 4. Numerical Warnings

The implementation generates multiple runtime warnings:

- `RuntimeWarning: invalid value encountered in log` in multiple locations
- Suggests potential issues with negative values being passed to `np.log()`

### 5. Missing or Simplified Components

Comparing the vectorized implementation to the scalar version reveals:

- Some functions appear to be simplified or missing components
- The `birk_tot()` function returns fewer values than expected
- Field components that exist in the scalar version may be missing

## Root Cause Analysis

The vectorized implementation in `direction_vectorize_6.md` appears to be either:

1. **Incomplete**: Missing critical components or calculations
2. **Incorrectly Translated**: Errors in converting scalar logic to vectorized operations
3. **Based on Different Version**: May be based on a different or simplified version of T01

## Recommendations

1. **Complete Rewrite Needed**: The current vectorized implementation needs a complete rewrite based directly on the scalar T01 implementation

2. **Incremental Approach**: Vectorize one function at a time and validate each against the scalar version

3. **Proper Testing**: Each vectorized function should be tested independently before integration

4. **Numerical Stability**: Pay special attention to division by zero and log of negative numbers

5. **Documentation**: The vectorization process should be documented step-by-step with validation results

## Conclusion

The T01 vectorized implementation provided in `direction_vectorize_6.md` is not production-ready and has fundamental issues that prevent it from producing accurate results. A complete rewrite following proper vectorization principles and thorough testing is required.