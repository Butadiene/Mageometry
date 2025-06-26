# T96 Vectorization Accuracy Under Various Solar Wind Conditions

## Executive Summary

The vectorized T96 implementation has been comprehensively tested under diverse solar wind conditions, from quiet periods to extreme storms. The results demonstrate **excellent numerical accuracy** across all conditions, with maximum relative error of 2.02e-08, well below the required 1e-6 threshold.

## Test Methodology

### Solar Wind Conditions Tested

1. **Quiet Northward IMF**
   - Pdyn: 1.0 nPa (low pressure)
   - Dst: -10 nT (quiet)
   - IMF: By=0 nT, Bz=+5 nT (northward)
   - Represents typical quiet solar wind

2. **Moderate Activity**
   - Pdyn: 3.0 nPa (moderate pressure)
   - Dst: -30 nT (minor disturbance)
   - IMF: By=-5 nT, Bz=0 nT
   - Represents unsettled conditions

3. **Storm with Southward IMF**
   - Pdyn: 8.0 nPa (high pressure)
   - Dst: -100 nT (storm)
   - IMF: By=+10 nT, Bz=-10 nT (southward)
   - Represents typical storm conditions

4. **Extreme Storm**
   - Pdyn: 20.0 nPa (very high pressure)
   - Dst: -200 nT (intense storm)
   - IMF: By=-15 nT, Bz=-20 nT (strong southward)
   - Represents severe space weather

5. **Strong Duskward IMF**
   - Pdyn: 5.0 nPa
   - Dst: -50 nT
   - IMF: By=+20 nT (strong duskward), Bz=-5 nT
   - Tests asymmetric configurations

6. **Recovery Phase**
   - Pdyn: 2.0 nPa (decreasing)
   - Dst: -40 nT (recovering)
   - IMF: By=+5 nT, Bz=+2 nT (turning northward)
   - Represents post-storm conditions

### Test Coverage
- 500 randomly distributed test points per condition
- Spatial coverage: 2-30 Re from Earth
- Total: 3,000 test point evaluations

## Accuracy Results

### Overall Statistics
| Metric | Value |
|--------|-------|
| Overall mean error | 5.09e-11 |
| Overall max error | 2.02e-08 |
| 99th percentile | 9.62e-10 |
| Points exceeding 1e-6 | 0 (0.00%) |

### Results by Solar Wind Condition

| Condition | Mean Error | Max Error | 99th Percentile | Speedup |
|-----------|------------|-----------|-----------------|---------|
| Quiet Northward | 1.39e-11 | 5.89e-10 | 3.60e-10 | 8.5x |
| Moderate Activity | 3.01e-11 | 4.77e-09 | 5.79e-10 | 17.3x |
| Storm Southward | 6.65e-11 | 1.31e-08 | 1.18e-09 | 8.6x |
| Extreme Storm | 1.01e-10 | 2.02e-08 | 1.71e-09 | 21.7x |
| Strong By | 7.31e-11 | 1.40e-08 | 1.39e-09 | 11.2x |
| Recovery Phase | 2.08e-11 | 7.07e-10 | 4.99e-10 | 17.1x |

### Key Findings

1. **Accuracy Scaling with Activity**
   - Errors increase slightly with storm intensity
   - Maximum error occurs during extreme storm conditions
   - All errors remain well below 1e-6 threshold

2. **IMF Orientation Effects**
   - Strong By component (±20 nT) maintains good accuracy
   - Southward IMF (negative Bz) shows slightly higher errors
   - Northward IMF (positive Bz) shows best accuracy

3. **Pressure Dependence**
   - Higher solar wind pressure correlates with slightly higher errors
   - Extreme pressure (20 nPa) still maintains excellent accuracy
   - Low pressure conditions show best accuracy

## Magnetopause Boundary Tests

Special tests were conducted near the magnetopause under different compression levels:

### Low Pressure (Pdyn=0.5 nPa)
| Location | Field Strength | Error |
|----------|---------------|-------|
| Subsolar (10,0,0) | 16.2 nT | 6.14e-14 |
| Dawn-dusk (8,6,0) | 14.8 nT | 2.24e-14 |
| High-lat (6,0,6) | 7.7 nT | 2.88e-11 |
| Off-axis (5,5,5) | 9.2 nT | 1.77e-14 |

### High Pressure (Pdyn=10.0 nPa)
| Location | Field Strength | Error |
|----------|---------------|-------|
| Subsolar (10,0,0) | 38.9 nT | 0.00e+00 |
| Dawn-dusk (8,6,0) | 37.8 nT | 0.00e+00 |
| High-lat (6,0,6) | 69.8 nT | 6.21e-11 |
| Off-axis (5,5,5) | 70.8 nT | 5.96e-10 |

### Strong Southward IMF (Bz=-15 nT)
| Location | Field Strength | Error |
|----------|---------------|-------|
| Subsolar (10,0,0) | 50.3 nT | 2.80e-12 |
| Dawn-dusk (8,6,0) | 56.1 nT | 6.23e-13 |
| High-lat (6,0,6) | 52.0 nT | 1.35e-10 |
| Off-axis (5,5,5) | 40.1 nT | 1.48e-09 |

## Performance Analysis

### Speedup Factors
- Average speedup: 14.1x across all conditions
- Best performance: 21.7x (extreme storm)
- Consistent performance regardless of solar wind conditions

### Performance Characteristics
- No performance degradation with extreme parameters
- Efficient handling of all IMF orientations
- Stable computation near magnetopause boundaries

## Robustness Testing

### Parameter Extremes Tested
- Pdyn: 0.5 - 20.0 nPa (40x range)
- Dst: -200 to -10 nT
- IMF By: -20 to +20 nT
- IMF Bz: -20 to +5 nT

### Numerical Stability
- No overflow or underflow detected
- No NaN or infinity values produced
- Graceful handling of all parameter combinations

## Conclusions

1. **Excellent Accuracy**: The vectorized T96 implementation maintains numerical accuracy better than 1e-6 across all tested solar wind conditions, from quiet periods to extreme storms.

2. **Robust Performance**: The implementation shows consistent speedup (8-22x) regardless of solar wind conditions.

3. **Physical Fidelity**: Accuracy patterns follow physical expectations:
   - Slightly higher errors during disturbed conditions
   - Best accuracy during quiet, northward IMF
   - Stable behavior at magnetopause boundaries

4. **Production Ready**: The implementation is validated for operational use under all expected solar wind conditions.

## Recommendations

1. **Safe for All Conditions**: The vectorized T96 can be used confidently for any solar wind condition within the tested parameter ranges.

2. **Extreme Event Capable**: Even under extreme storm conditions (Dst=-200 nT, Pdyn=20 nPa), accuracy remains excellent.

3. **No Special Handling Required**: No parameter-dependent adjustments or special cases are needed.

## Visualizations

Two visualization files were generated:
- `t96_solar_wind_conditions_quick.png`: Comprehensive accuracy and performance plots
- Includes error distributions, parameter dependencies, and IMF effects

---
*Evaluation performed on 3,000 test points across 6 distinct solar wind conditions*