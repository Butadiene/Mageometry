# IGRF Implementation Analysis

## Overview

The IGRF (International Geomagnetic Reference Field) implementation in geopack consists of several interconnected functions that calculate the Earth's internal magnetic field. This analysis identifies the key components and considerations for vectorization.

## Call Chain

### 1. Entry Points
- `igrf_gsw(xgsw, ygsw, zgsw)` - GSW coordinates
- `igrf_gsm(xgsm, ygsm, zgsm)` - GSM coordinates  

### 2. Core Calculation
- `igrf_geo(r, theta, phi)` - Spherical geographic coordinates

### 3. Supporting Functions
- `gswgsm()` - Coordinate transformation GSW ↔ GSM
- `geogsm()` - Coordinate transformation GEO ↔ GSM
- `sphcar()` - Coordinate transformation Spherical ↔ Cartesian
- `bspcar()` - Field component transformation Spherical → Cartesian

## Global Variables

The IGRF implementation relies on several global variables that must be handled carefully:

### 1. IGRF Coefficients (set by `init_igrf()` and `load_igrf()`)
- `g` - Schmidt-normalized g coefficients (array of 105 elements)
- `h` - Schmidt-normalized h coefficients (array of 105 elements)
- `rec` - Recursion coefficients for Legendre polynomials (array of 105 elements)

### 2. Transformation Matrices (set by `recalc()`)
- `a11, a21, a31, a12, a22, a32, a13, a23, a33` - GEO to GSM transformation matrix
- `e11, e21, e31, e12, e22, e32, e13, e23, e33` - GSM to GSW transformation matrix

### 3. Other Global State
- `igrf` - Dictionary containing time-dependent coefficient arrays
- `nmn, mns, nyear, years, yruts` - IGRF metadata

## Core Algorithm (`igrf_geo`)

The `igrf_geo` function implements the spherical harmonic expansion:

```python
def igrf_geo(r, theta, phi):
    # 1. Calculate maximal order based on radial distance
    nm = min(13, 3 + 30/(r+2))
    
    # 2. Precompute radial functions
    a[n] = (1/r)^(n+2)
    b[n] = (n+1) * a[n]
    
    # 3. Calculate Legendre polynomials recursively
    # For m=0:
    P(n,0) = ct*P(n-1,0) - K(n,0)*P(n-2,0)
    
    # For m>0:
    P(n,m) = ct*P(n-1,m) - K(n,m)*P(n-2,m)
    P(m,m) = st*P(m-1,m-1)
    
    # 4. Sum spherical harmonics
    Br = Σ b[n] * (g[n,m]*cos(m*φ) + h[n,m]*sin(m*φ)) * P(n,m)
    Bθ = -Σ a[n] * (g[n,m]*cos(m*φ) + h[n,m]*sin(m*φ)) * dP(n,m)/dθ
    Bφ = Σ a[n] * m * (g[n,m]*sin(m*φ) - h[n,m]*cos(m*φ)) * P(n,m)/sin(θ)
```

## Vectorization Challenges

### 1. Global Variable Dependencies
- Functions depend on global arrays `g`, `h`, `rec` set by `recalc()`
- Transformation matrices are also global
- Need to pass these as parameters or ensure thread-safe access

### 2. Recursive Calculations
- Legendre polynomials use recursive formulas
- Current implementation uses sequential loops
- Need to vectorize the recursion relations

### 3. Conditional Logic
- Special handling at poles (`smlst` flag when `sin(theta) < 1e-5`)
- Variable loop bounds based on `nm` (depends on `r`)
- Need to handle with `np.where()` or masked operations

### 4. Index Management
- Complex indexing scheme: `mn = n*(n+1)/2 + m`
- Maps 2D (n,m) indices to 1D array index
- Need to vectorize index calculations

## Vectorization Strategy

### 1. Input Handling
```python
# Ensure arrays
r = np.atleast_1d(r)
theta = np.atleast_1d(theta) 
phi = np.atleast_1d(phi)
```

### 2. Precompute Arrays
```python
# Vectorize radial functions
a = np.zeros((r.shape[0], k))
b = np.zeros((r.shape[0], k))
ar = 1/r
a[:, 0] = ar * ar
b[:, 0] = a[:, 0]
for n in range(1, k):
    a[:, n] = a[:, n-1] * ar
    b[:, n] = a[:, n] * (n+1)
```

### 3. Vectorize Trigonometric Functions
```python
ct = np.cos(theta)
st = np.sin(theta)
# Precompute sin(m*phi), cos(m*phi) for all m
sin_mphi = np.sin(np.outer(phi, np.arange(k)))
cos_mphi = np.cos(np.outer(phi, np.arange(k)))
```

### 4. Handle Pole Singularity
```python
smlst = np.abs(st) < 1e-5
# Use masked arrays or np.where for conditional calculations
```

### 5. Vectorize Legendre Polynomial Recursion
- Most challenging part
- Consider computing P(n,m) for all points simultaneously
- May need to reshape arrays for broadcasting

## Required Helper Functions to Vectorize

1. `sphcar_vectorized()` - Already exists in some form
2. `bspcar_vectorized()` - Field component transformation
3. `geogsm_vectorized()` - Coordinate transformation
4. `gswgsm_vectorized()` - Coordinate transformation

## Testing Requirements

1. **Accuracy**: Compare with scalar version for various (r, theta, phi)
2. **Edge Cases**: 
   - Poles (theta ≈ 0 or π)
   - Different radial distances (affects `nm`)
   - Full range of phi values
3. **Performance**: Benchmark with arrays of different sizes
4. **Consistency**: Ensure proper handling of global state

## Implementation Priority

1. Start with core `igrf_geo` vectorization
2. Then vectorize coordinate transformations
3. Finally create vectorized `igrf_gsm` and `igrf_gsw`
4. Ensure backward compatibility with scalar inputs

## Test Vectors

From `test_geopack1.py`, with test position in GSM coordinates (1, 2, 3):
- `igrf_gsm(1, 2, 3)` → `(262.829, -19.306, -50.346)` nT
- `igrf_gsw(...)` → `(263.870, -20.600, -43.992)` nT

These values provide validation targets for the vectorized implementation.

## Existing Vectorized Functions

The following coordinate transformation functions already have vectorized versions:
- `geogsm_vectorized()` in `coordinates_vectorized.py`
- `gswgsm_vectorized()` in `coordinates_vectorized.py` 
- `sphcar_vectorized()` in `coordinates_vectorized_complex.py`
- `bspcar_vectorized()` in `coordinates_vectorized_complex.py`

These can be used directly in the IGRF vectorization.