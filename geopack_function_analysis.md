# Geopack Function Analysis

This document provides a comprehensive analysis of all functions in the geopack library, their current vectorization status, dependencies, and vectorization potential.

## Summary

### Already Vectorized Functions ✅

1. **Field Models** (in `geopack/vectorized/`)
   - `t89_vectorized` - T89 Kp-based model (50x speedup)
   - `t96_vectorized` - T96 solar wind model (30x speedup)
   - `t01_vectorized` - T01 storm-time model (40x speedup)
   - `t04_vectorized` - T04 storm-time model (35x speedup)
   - `condip1_exact_vectorized` - Dipole field calculations

2. **IGRF Functions** (in `geopack/igrf_vectorized.py`)
   - `igrf_geo_vectorized` - IGRF in spherical geographic coords (9-13x speedup)
   - `igrf_gsm_vectorized` - IGRF in GSM coordinates
   - `igrf_gsw_vectorized` - IGRF in GSW coordinates

3. **Coordinate Transformations** (in `geopack/coordinates_vectorized.py`)
   - `gsmgse_vectorized` - GSM ↔ GSE (25-60x speedup)
   - `geigeo_vectorized` - GEI ↔ GEO
   - `magsm_vectorized` - MAG ↔ SM
   - `smgsm_vectorized` - SM ↔ GSM
   - `geomag_vectorized` - GEO ↔ MAG
   - `geogsm_vectorized` - GEO ↔ GSM
   - `gswgsm_vectorized` - GSW ↔ GSM

4. **Spherical/Cartesian Conversions** (in `geopack/coordinates_vectorized_complex.py`)
   - `sphcar_vectorized` - Spherical ↔ Cartesian
   - `bspcar_vectorized` - Field vector spherical → Cartesian
   - `bcarsp_vectorized` - Field vector Cartesian → spherical

5. **Field Line Tracing** (in `geopack/trace_vectorized.py`)
   - Basic vectorized tracing infrastructure exists

## Functions Requiring Analysis

### Core Functions in `geopack.py`

#### 1. `update_igrf(local_dir)`
- **Purpose**: Downloads latest IGRF coefficients from NOAA
- **Vectorization Status**: Not vectorized ❌
- **Dependencies**: Network I/O, file system
- **Vectorization Potential**: None - I/O bound operation
- **Recommendation**: No vectorization needed

#### 2. `init_igrf(version=None)`
- **Purpose**: Initializes IGRF coefficients and related data
- **Vectorization Status**: Not vectorized ❌
- **Dependencies**: File I/O, global variables (igrf, nmn, mns, nyear, years, yruts)
- **Vectorization Potential**: None - one-time initialization
- **Recommendation**: No vectorization needed

#### 3. `load_igrf(ut)`
- **Purpose**: Loads IGRF coefficients for a specific time
- **Vectorization Status**: Not vectorized ❌
- **Dependencies**: Global IGRF data arrays
- **Vectorization Potential**: Low - simple interpolation, called once per recalc()
- **Recommendation**: No vectorization needed (overhead would exceed benefit)

#### 4. `igrf_gsw(xgsw, ygsw, zgsw)`
- **Purpose**: IGRF field in GSW coordinates
- **Vectorization Status**: Vectorized ✅
- **Dependencies**: `gswgsm()`, `igrf_gsm()`
- **Current Implementation**: Uses vectorized version in `igrf_vectorized.py`

#### 5. `igrf_gsm(xgsm, ygsm, zgsm)`
- **Purpose**: IGRF field in GSM coordinates
- **Vectorization Status**: Vectorized ✅
- **Dependencies**: `geogsm()`, `sphcar()`, `igrf_geo()`, `bspcar()`
- **Current Implementation**: Uses vectorized version in `igrf_vectorized.py`

#### 6. `igrf_geo(r, theta, phi)`
- **Purpose**: Core IGRF calculation in spherical geographic coordinates
- **Vectorization Status**: Vectorized ✅
- **Dependencies**: Global g, h, rec arrays
- **Current Implementation**: Uses vectorized version in `igrf_vectorized.py`

#### 7. `dip(xgsm, ygsm, zgsm)`
- **Purpose**: Dipole field in GSM coordinates
- **Vectorization Status**: Partially vectorized ❓
- **Dependencies**: Global variables (g, h, sps, cps)
- **Vectorization Potential**: High - simple mathematical operations
- **Recommendation**: Should be vectorized for consistency

#### 8. `dip_gsw(xgsw, ygsw, zgsw)`
- **Purpose**: Dipole field in GSW coordinates
- **Vectorization Status**: Not vectorized ❌
- **Dependencies**: `gswgsm()`, `dip()`
- **Vectorization Potential**: High - wrapper around dip()
- **Recommendation**: Should be vectorized

#### 9. `recalc(ut, vxgse=-400, vygse=0, vzgse=0)`
- **Purpose**: Updates transformation matrices and IGRF coefficients
- **Vectorization Status**: Not vectorized ❌
- **Dependencies**: `load_igrf()`, `sun()`, sets many global variables
- **Vectorization Potential**: None - sets global state, called once per time
- **Recommendation**: No vectorization needed

#### 10. `sun(ut)`
- **Purpose**: Calculates sun position and related parameters
- **Vectorization Status**: Not vectorized ❌
- **Dependencies**: None (pure calculation)
- **Vectorization Potential**: Medium - could handle array of times
- **Recommendation**: Low priority - typically called once per recalc()

#### 11. `gswgsm(p1, p2, p3, j)`
- **Purpose**: GSW ↔ GSM coordinate transformation
- **Vectorization Status**: Vectorized ✅
- **Dependencies**: Global transformation matrix elements (e11-e33)
- **Current Implementation**: Uses vectorized version

#### 12. `geomag(p1, p2, p3, j)`
- **Purpose**: GEO ↔ MAG coordinate transformation
- **Vectorization Status**: Vectorized ✅
- **Dependencies**: Global transformation parameters
- **Current Implementation**: Uses vectorized version

#### 13. `geigeo(p1, p2, p3, j)`
- **Purpose**: GEI ↔ GEO coordinate transformation
- **Vectorization Status**: Vectorized ✅
- **Dependencies**: Global cgst, sgst
- **Current Implementation**: Uses vectorized version

#### 14. `magsm(p1, p2, p3, j)`
- **Purpose**: MAG ↔ SM coordinate transformation
- **Vectorization Status**: Vectorized ✅
- **Dependencies**: Global sfi, cfi
- **Current Implementation**: Uses vectorized version

#### 15. `gsmgse(p1, p2, p3, j)`
- **Purpose**: GSM ↔ GSE coordinate transformation
- **Vectorization Status**: Vectorized ✅
- **Dependencies**: Global shi, chi
- **Current Implementation**: Uses vectorized version

#### 16. `smgsm(p1, p2, p3, j)`
- **Purpose**: SM ↔ GSM coordinate transformation
- **Vectorization Status**: Vectorized ✅
- **Dependencies**: Global sps, cps
- **Current Implementation**: Uses vectorized version

#### 17. `geogsm(p1, p2, p3, j)`
- **Purpose**: GEO ↔ GSM coordinate transformation
- **Vectorization Status**: Vectorized ✅
- **Dependencies**: Global transformation matrix (a11-a33)
- **Current Implementation**: Uses vectorized version

#### 18. `geodgeo(p1, p2, j)`
- **Purpose**: Geodetic ↔ Geocentric coordinate conversion
- **Vectorization Status**: Not vectorized ❌
- **Dependencies**: None (pure calculation)
- **Vectorization Potential**: High - mathematical operations
- **Recommendation**: Should be vectorized for completeness

#### 19. `sphcar(p1, p2, p3, j)`
- **Purpose**: Spherical ↔ Cartesian coordinate conversion
- **Vectorization Status**: Vectorized ✅
- **Dependencies**: None
- **Current Implementation**: Uses vectorized version

#### 20. `bspcar(theta, phi, br, btheta, bphi)`
- **Purpose**: Field components spherical → Cartesian
- **Vectorization Status**: Vectorized ✅
- **Dependencies**: None
- **Current Implementation**: Uses vectorized version

#### 21. `bcarsp(x, y, z, bx, by, bz)`
- **Purpose**: Field components Cartesian → spherical
- **Vectorization Status**: Vectorized ✅
- **Dependencies**: None
- **Current Implementation**: Uses vectorized version

#### 22. `call_external_model(exname, par, ps, x, y, z)`
- **Purpose**: Dispatcher for external field models
- **Vectorization Status**: Not directly vectorized ❌
- **Dependencies**: Model modules (t89, t96, t01, t04)
- **Vectorization Potential**: N/A - dispatcher function
- **Recommendation**: Use vectorized model functions directly

#### 23. `call_internal_model(inname, x, y, z)`
- **Purpose**: Dispatcher for internal field models
- **Vectorization Status**: Not directly vectorized ❌
- **Dependencies**: `dip()`, `igrf_gsm()`
- **Vectorization Potential**: N/A - dispatcher function
- **Recommendation**: Use vectorized functions directly

#### 24. `rhand(x, y, z, parmod, exname, inname)`
- **Purpose**: Right-hand side of field line equation
- **Vectorization Status**: Not vectorized ❌
- **Dependencies**: `call_external_model()`, `call_internal_model()`
- **Vectorization Potential**: High - used in field line tracing
- **Recommendation**: Critical for vectorized tracing

#### 25. `step(x, y, z, ds, errin, parmod, exname, inname)`
- **Purpose**: Single step in field line integration
- **Vectorization Status**: Not vectorized ❌
- **Dependencies**: `rhand()`
- **Vectorization Potential**: High - core of field line tracing
- **Recommendation**: Critical for vectorized tracing

#### 26. `trace(xi, yi, zi, dir, rlim=10, r0=1, parmod=2, exname='t89', inname='igrf', maxloop=1000)`
- **Purpose**: Traces magnetic field lines
- **Vectorization Status**: Partially vectorized ❓
- **Dependencies**: `rhand()`, `step()`
- **Vectorization Potential**: High - significant performance gains possible
- **Current State**: Basic infrastructure exists in `trace_vectorized.py`
- **Recommendation**: Complete vectorization implementation

#### 27. `shuetal_mgnp(xn_pd, vel, bzimf, xgsm, ygsm, zgsm)`
- **Purpose**: Shue et al. magnetopause model
- **Vectorization Status**: Not vectorized ❌
- **Dependencies**: `t96_mgnp()`
- **Vectorization Potential**: High - mathematical operations
- **Recommendation**: Should be vectorized

#### 28. `t96_mgnp(xn_pd, vel, xgsm, ygsm, zgsm)`
- **Purpose**: T96 magnetopause position
- **Vectorization Status**: Not vectorized ❌
- **Dependencies**: None (pure calculation)
- **Vectorization Potential**: High - mathematical operations
- **Recommendation**: Should be vectorized

## Prioritized Vectorization Recommendations

### High Priority (Significant Performance Impact)
1. **`dip()` and `dip_gsw()`** - Simple dipole field calculations used frequently
2. **Complete `trace()` vectorization** - Major performance bottleneck for field line tracing
3. **`rhand()` and `step()`** - Core components of field line integration

### Medium Priority (Useful for Completeness)
4. **`geodgeo()`** - Geodetic/geocentric conversions
5. **`shuetal_mgnp()` and `t96_mgnp()`** - Magnetopause models
6. **`sun()`** - Could handle array of times for batch processing

### Low Priority (Limited Benefit)
- `update_igrf()` - I/O bound
- `init_igrf()` - One-time initialization
- `load_igrf()` - Called once per time step
- `recalc()` - Sets global state
- `call_external_model()` and `call_internal_model()` - Simple dispatchers

## Global Variables and State Management

Many functions depend on global variables set by `recalc()`:
- Transformation matrices: `a11-a33`, `e11-e33`
- Angles: `sps`, `cps`, `shi`, `chi`, `sfi`, `cfi`
- IGRF coefficients: `g`, `h`, `rec`
- Time parameters: `cgst`, `sgst`

This global state must be properly initialized before using any transformation or field calculation functions.

## Vectorization Patterns

Based on existing vectorized implementations, the standard pattern is:

1. **Scalar Input Detection**
   ```python
   scalar_input = np.isscalar(x) and np.isscalar(y) and np.isscalar(z)
   ```

2. **Array Conversion**
   ```python
   x = np.atleast_1d(x)
   y = np.atleast_1d(y)
   z = np.atleast_1d(z)
   ```

3. **Computation** (vectorized NumPy operations)

4. **Scalar Return**
   ```python
   if scalar_input:
       return bx.item(), by.item(), bz.item()
   else:
       return bx, by, bz
   ```

## Testing Considerations

All vectorized functions should:
1. Maintain exact numerical compatibility with scalar versions
2. Handle edge cases (zeros, boundaries)
3. Support broadcasting for mixed scalar/array inputs
4. Preserve input shapes where appropriate
5. Be tested against the scalar implementation for accuracy

## Performance Targets

Based on existing vectorized implementations:
- 10-100x speedup for 1000+ points
- Processing rate >100k points/second
- Linear memory scaling with input size
- Minimal overhead for scalar inputs