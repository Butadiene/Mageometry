# T01 Vectorization Complete History

This document consolidates the complete history of T01 model vectorization attempts, challenges, and solutions.

## Overview

The T01 (Tsyganenko 2001) magnetospheric field model vectorization was a complex process involving multiple iterations, debugging sessions, and fixes. This document preserves the complete history for future reference.

## Vectorization Timeline

### Phase 1: Initial Implementation (direction_vectorize_6.md - direction_vectorize_9.md)
- Initial vectorization of T01 components
- Discovery of component calculation issues
- Problems with ring current and Birkeland current implementations

### Phase 2: Component Debugging (direction_vectorize_10.md - direction_vectorize_15.md)
- Systematic isolation of each field component
- Discovery of issues in:
  - Ring current shielding calculations
  - Birkeland current field assembly
  - Tail field mode calculations
  - Interconnection field implementation

### Phase 3: Numerical Issues (direction_vectorize_16.md - direction_vectorize_20.md)
- Identification of numerical instabilities
- Problems with extreme parameter values
- Division by zero issues near origin
- Exponential overflow in various components

### Phase 4: Final Fixes (direction_vectorize_21.md - direction_vectorize_27.md)
- Implementation of exponential clipping
- Fixes for IMF penetration terms
- Resolution of most numerical stability issues

## Key Issues Encountered

### 1. Ring Current Shield Calculation
- **Problem**: Incorrect implementation of coordinate transformations
- **Solution**: Fixed rotation angles and coordinate scaling
- **Files**: ring_current_vectorized.py, particularly rc_shield_vectorized function

### 2. Exponential Overflow
- **Problem**: Unclipped exponentials causing overflow with extreme parameters
- **Location**: Tail harmonics (shlcar5x5_vectorized) and ring current calculations
- **Solution**: Added clipping: `np.clip(x * sqpr, -740.0, 88.0)`

### 3. IMF Penetration Term
- **Initial confusion**: Whether to scale hyimf/hzimf by factimf
- **Resolution**: The scalar Python implementation was correct; no change needed
- **Note**: Original Fortran may have had different implementation

### 4. Coordinate System Issues
- **Problem**: Inconsistent use of scaled vs unscaled coordinates
- **Solution**: Careful tracking of when to use xx,yy,zz vs x,y,z

### 5. Numerical Stability Near Origin
- **Problem**: Division by zero and invalid values for r < 1e-5
- **Solution**: Added origin_mask to handle points near origin specially

## Remaining Issues

Despite extensive debugging, some discrepancies remain:

### 1. Large Errors at Flanks
- Dawn/dusk flanks show ~27-28 nT errors
- Appears to be systematic, not random
- Symmetric pattern suggests coordinate transformation issue

### 2. By Component Discrepancy
- At (0,0,1): Scalar By=52.249, Vector By=-3.592
- 55.8 nT difference with opposite signs
- Suggests possible sign error or missing term

### 3. Numerical Issues Near Y-axis
- Errors increase dramatically near Y-axis at small r
- Maximum error of 55.8 nT at (0,0,1)
- Related to numerical instabilities in field calculations

## Test Results Summary

### Extreme Parameter Tests
- Config 6 (pdyn=25, Dst=-300): max error reduced from 10.8 to 1.7 nT
- High-pressure strong IMF: max error reduced from 17.9 to 2.4 nT
- Improvements due to exponential clipping fixes

### Error Pattern Analysis
- Errors concentrated at:
  - Y=0 axis (especially small Z)
  - Dawn/dusk flanks (Y=±8)
  - Ring current region
- Errors show clear spatial pattern, not random

## Implementation Details

### Key Functions
1. **t01_vectorized**: Main entry point
2. **extall_vectorized**: Field assembly and region determination
3. **full_rc_vectorized**: Ring current with shielding
4. **birk_tot_vectorized**: Birkeland current fields
5. **deformed_vectorized**: Tail field with warping/bending

### Critical Algorithms
1. **iterate_sigma_vectorized**: Iterative coordinate unwarping
2. **rc_shield_vectorized**: Ring current shielding calculation
3. **shlcar5x5_vectorized**: 5x5 Cartesian harmonic shield

## Lessons Learned

1. **Vectorization Complexity**: T01's iterative algorithms and complex coordinate transformations make vectorization challenging

2. **Numerical Stability**: Essential to protect all exponentials and divisions

3. **Testing Strategy**: Component isolation was crucial for debugging

4. **Documentation**: Maintaining detailed progress logs was invaluable

5. **Validation**: Need comprehensive tests with extreme parameters

## Future Work

1. **Resolve Flank Errors**: Investigate the ~27 nT errors at dawn/dusk flanks

2. **Fix By Component**: Debug the sign/calculation issue in By component

3. **Improve Near-Origin Handling**: Better numerical stability for small r

4. **Performance Optimization**: Current implementation prioritizes correctness over speed

## File Organization

### Implementation Files
- `geopack/t01_vectorized.py`: Main T01 vectorized implementation
- `geopack/ring_current_vectorized.py`: Ring current components
- `geopack/birkeland_vectorized.py`: Birkeland current fields

### Documentation
- `docs/vectorization/direction_vectorize_*.md`: Progress logs
- `docs/vectorization/T01_VECTORIZATION_POLICY.md`: Implementation guidelines
- Various status and summary documents

### Key Test Files
- `tests/test_t01_extreme_params.py`: Extreme parameter validation
- `tests/analyze_error_pattern.py`: Spatial error analysis
- `tests/check_origin_handling.py`: Near-origin numerical issues

## Conclusion

The T01 vectorization project successfully implemented a vectorized version that handles most cases correctly, with significant improvements in numerical stability for extreme parameters. However, some systematic errors remain that require further investigation, particularly in the By component calculation and at the dawn/dusk flanks.

The extensive documentation and test suite provide a solid foundation for future improvements and debugging efforts.