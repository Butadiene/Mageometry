# T96 Vectorization Accuracy Report

## Executive Summary

The vectorized T96 implementation has been thoroughly evaluated against the scalar reference implementation across a comprehensive parameter space. The results demonstrate **excellent numerical accuracy** with all relative errors below 1e-6.

## Key Findings

### Accuracy Metrics

- **Mean relative error**: 3.97e-11
- **Median relative error**: 6.77e-16
- **Maximum relative error**: 1.80e-08
- **99th percentile error**: 7.64e-10
- **95th percentile error**: 1.00e-10

### Error Distribution

| Error Threshold | Points Exceeding | Percentage |
|----------------|------------------|------------|
| > 1e-10        | 501              | 5.01%      |
| > 1e-08        | 6                | 0.06%      |
| > 1e-06        | 0                | 0.00%      |
| > 1e-04        | 0                | 0.00%      |
| > 1e-02        | 0                | 0.00%      |

## Test Coverage

The evaluation covered 10,000 test points across:

### Spatial Regions
- **Near-Earth**: 1.5-3 Re (100 special cases)
- **Mid-field**: 3-10 Re
- **Far-field**: 10-30 Re
- **Magnetotail**: x = -50 to -10 Re (100 special cases)
- **High-latitude**: z = ±10 to ±20 Re (100 special cases)

### Parameter Space
- **Pdyn**: 0.5-10.0 nPa (solar wind pressure)
- **Dst**: -200 to +50 nT (storm-time index)
- **ByIMF**: -10 to +10 nT (IMF Y-component)
- **BzIMF**: -10 to +10 nT (IMF Z-component)
- **Tilt angle**: -28.6° to +28.6°

## Regional Analysis

### Error by Distance from Earth

| Region (Re) | Mean Error | Max Error |
|-------------|------------|-----------|
| 0-3         | 6.79e-11   | 4.39e-09  |
| 3-10        | 4.53e-11   | 5.64e-09  |
| 10-20       | 2.58e-11   | 4.93e-09  |
| 20-50       | 4.56e-11   | 1.80e-08  |

### Error by IMF Conditions

| Condition            | Mean Error | Max Error |
|---------------------|------------|-----------|
| Northward IMF (Bz>0)| 2.08e-11   | 1.10e-08  |
| Southward IMF (Bz<0)| 5.90e-11   | 1.80e-08  |
| Strong By (\|By\|>5)| 4.76e-11   | 1.51e-08  |

## Worst Case Analysis

The point with highest relative error:
- **Position**: (-45.735, 5.020, -0.674) Re
- **Distance**: 46.015 Re (far magnetotail)
- **Parameters**: Pdyn=7.78 nPa, Dst=-75.9 nT, ByIMF=-0.27 nT, BzIMF=-3.47 nT
- **Tilt**: -15.3°
- **Field values**: 
  - Scalar: (6.551, -0.292, 2.461) nT
  - Vector: (6.551, -0.292, 2.461) nT
- **Relative error**: 1.80e-08

This worst case still maintains excellent accuracy with differences at the level of numerical precision.

## Performance Benchmarks

### Single Point Calculations
- **Scalar implementation**: 1,311 points/sec
- **Vectorized (single calls)**: 254 points/sec
- **Overhead factor**: 0.2x (due to array operations overhead for single points)

### Batch Processing
- **Scalar (1000 points)**: 0.763 seconds
- **Vectorized batch**: 0.025 seconds
- **Speedup**: **30.1x**
- **Throughput**: 39,506 points/sec

## Implementation Quality

### Numerical Stability
- No instances of numerical overflow or underflow
- Proper handling of edge cases (zero fields, boundary conditions)
- Consistent behavior across all parameter ranges

### Interface Compatibility
- Single point interface matches scalar exactly (error < 1e-13)
- Preserves scalar input/output behavior
- Full backward compatibility maintained

## Conclusions

1. **Accuracy**: The vectorized T96 implementation achieves numerical accuracy well within acceptable limits for scientific applications. The maximum relative error of 1.80e-08 is negligible for practical magnetospheric modeling.

2. **Performance**: The 30x speedup for batch processing makes large-scale simulations feasible while maintaining accuracy.

3. **Robustness**: The implementation handles extreme parameter values and edge cases correctly, with no numerical instabilities observed.

4. **Recommendation**: The vectorized T96 implementation is ready for production use in scientific applications requiring high-performance magnetospheric field calculations.

## Technical Notes

- All component functions (dipole, shielding, tail/ring currents, Birkeland currents, interconnection field) have been successfully vectorized
- The implementation uses pure NumPy with no external dependencies
- Memory usage scales linearly with input size
- The code maintains the exact mathematical formulation of the original T96 model

---
*Evaluation performed on 10,000 test points covering the full T96 parameter space*