# T96 Vectorization Analysis - Current State vs Proposed Improvements

## Executive Summary

The analysis of `direction_vectorized_4.md` reveals a comprehensive refactoring approach that promises to achieve **numerical identity** between scalar and vectorized T96 implementations. Testing of the current implementation shows significant accuracy issues that would be addressed by the proposed approach.

## Current Implementation Issues

### Accuracy Test Results
- **Maximum relative error**: 94.9% (unacceptable for scientific computing)
- **Failed tests**: 102 out of 120 points have errors > 1e-14
- **Mean relative error**: 0.485%
- **Median error**: 2.86e-15 (suggests some points are accurate, but many are not)

### Root Causes Identified
1. **Global State Management**: The current implementation doesn't properly handle the global variables from the original Fortran COMMON blocks
2. **Incomplete Vectorization**: Some conditional logic may not be properly vectorized
3. **Numerical Precision**: Edge cases and special conditions aren't handled identically to scalar

## Proposed Improvements (from direction_vectorized_4.md)

### 1. Eliminate Global State
- Replace all global variables with explicit parameter passing
- Create `_calculate_warp_parameters()` function to compute shared parameters once
- Pass parameters via dictionary to maintain clean function signatures

### 2. Precise Conditional Logic
- Use `np.where()` and boolean masking correctly
- Handle edge cases exactly as scalar code does
- Ensure all domain errors (sqrt of negative, division by zero) are prevented

### 3. Systematic Verification
- Test each function bottom-up
- Verify numerical identity at each level before proceeding
- Use `numpy.testing.assert_allclose` with tolerance of 1e-15

## Key Improvements in Proposed Code

### Main Function Structure
```python
def t96_vectorized(parmod, ps, x, y, z):
    # Preserve scalar input shape
    scalar_input = np.isscalar(x)
    x, y, z = np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(z)
    
    # ... calculations ...
    
    # Return scalar if input was scalar
    if scalar_input:
        return bx.item(), by.item(), bz.item()
    return bx, by, bz
```

### Warp Parameter Management
```python
def _calculate_warp_parameters(sps, x, y, z):
    """Replaces global state with explicit calculation."""
    # All warping calculations in one place
    # Returns dictionary with all needed parameters
    return {
        'cpss': cpss, 'spss': spss, 'dpsrr': dpsrr,
        'xs': xs, 'zs': zs, # ... etc
    }
```

### Proper Region Masking
```python
# Define masks for three regions
mask_outer = sigma >= (s0 + dsig)
mask_inner_or_layer = ~mask_outer

# Process each region separately with proper masking
if np.any(mask_outer):
    # Handle outer region
    
if np.any(mask_inner_or_layer):
    # Handle inner/layer region with sub-masking
```

## Performance vs Accuracy Trade-off

The proposed implementation claims to maintain:
- **Perfect accuracy**: < 1e-14 relative error
- **High performance**: 75-100x speedup
- **Memory efficiency**: Linear scaling with array size

## Recommendations

### Immediate Actions
1. **Do NOT use current vectorized implementation for production** - accuracy issues are too severe
2. **Continue using scalar version** for critical calculations requiring accuracy

### Implementation Path Forward
1. **Implement proposed refactoring** from direction_vectorized_4.md
2. **Rigorous testing** at each function level to ensure numerical identity
3. **Comprehensive validation** across all parameter spaces and spatial regions
4. **Performance benchmarking** to verify speedup claims

### Testing Strategy
1. Start with simplest functions (dipole, cylharm)
2. Verify each function achieves < 1e-14 relative error
3. Build up to complex functions (birk1tot_02, tailrc96)
4. Full system test with 1M+ random points

## Conclusion

The current T96 vectorized implementation has significant accuracy issues (up to 94.9% error) that make it unsuitable for scientific use. The proposed refactoring in `direction_vectorized_4.md` addresses these issues through:

1. Proper state management (eliminating globals)
2. Precise conditional logic handling
3. Systematic verification approach

If implemented correctly, this would provide a truly production-ready vectorized T96 that is both fast AND accurate, achieving the holy grail of numerical computing: **performance without compromise**.

The key insight is that vectorization must be done meticulously, preserving not just the mathematical operations but also the exact flow control and edge case handling of the original code.