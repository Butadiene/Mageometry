# Field Line Directional Derivatives - Correct Formulation

## Overview

This document describes the correct implementation of the 9 directional derivative formulas for magnetic field line geometry in the Frenet-Serret frame.

## Mathematical Framework

### Frenet-Serret Frame
For a space curve (magnetic field line), we define the orthonormal frame:
- **T**: Unit tangent vector (along the field line), |T| = 1
- **n**: Unit principal normal vector (direction of curvature), |n| = 1
- **b**: Unit binormal vector (b = T × n), |b| = 1

These vectors form an orthonormal basis:
- T · T = n · n = b · b = 1
- T · n = T · b = n · b = 0

### The 9 Key Formulas

The directional derivatives of the frame vectors satisfy specific relationships. Here are the 9 formulas, 8 of which have non-zero values:

#### 1. Tangential Derivatives (∂/∂T)
These are the classical Frenet-Serret formulas:
- **(∂T/∂T)·n = κ** (curvature)
- **(∂T/∂T)·b = 0**
- **(∂n/∂T)·b = τ** (torsion)

With antisymmetry relations:
- (∂T/∂T)·n = -(∂n/∂T)·T = κ
- (∂T/∂T)·b = (∂b/∂T)·T = 0
- (∂n/∂T)·b = -(∂b/∂T)·n = τ

#### 2. Normal Derivatives (∂/∂n)
- **(∂T/∂n)·n**
- **(∂T/∂n)·b**
- **(∂n/∂n)·b**

With antisymmetry relations:
- (∂T/∂n)·n = -(∂n/∂n)·T
- (∂T/∂n)·b = -(∂b/∂n)·T
- (∂n/∂n)·b = -(∂b/∂n)·n

#### 3. Binormal Derivatives (∂/∂b)
- **(∂n/∂b)·b**
- **(∂n/∂b)·T**
- **(∂b/∂b)·T**

With antisymmetry relations:
- (∂n/∂b)·b = -(∂b/∂b)·n
- (∂n/∂b)·T = -(∂T/∂b)·n
- (∂b/∂b)·T = -(∂T/∂b)·b

### Important Properties

1. **Unit vector constraint**: Since T, n, and b are unit vectors (|T| = |n| = |b| = 1), their derivatives are perpendicular to themselves:
   - (∂T/∂s)·T = 0 for any direction s
   - (∂n/∂s)·n = 0 for any direction s
   - (∂b/∂s)·b = 0 for any direction s
   
   This is because d(U·U)/ds = 2U·(∂U/∂s) = 0, so U·(∂U/∂s) = 0.

2. **Self-components vanish**: All combinations like (∂T/∂T)·T, (∂n/∂n)·n, (∂b/∂b)·b are zero.

3. **Antisymmetry**: If (∂A/∂B)·C = value, then (∂C/∂B)·A = -value

4. **Other combinations vanish**: Combinations not listed in the 9 formulas (like (∂n/∂T)·n) are zero.

## Implementation

### Core Function

```python
def field_line_directional_derivatives_vectorized(model_func, parmod, ps, x, y, z, delta=0.01):
    """
    Calculate all 9 directional derivative formulas.
    
    Returns
    -------
    derivatives : dict
        Contains all 9 values:
        - 'dT_dT_n': κ (curvature)
        - 'dT_dT_b': 0
        - 'dn_dT_b': τ (torsion)
        - 'dT_dn_n', 'dT_dn_b', 'dn_dn_b'
        - 'dn_db_b', 'dn_db_T', 'db_db_T'
    """
```

### Numerical Method

We use finite differences to approximate directional derivatives:

1. For (∂A/∂B)·C:
   - Step forward and backward in the B direction
   - Calculate A at both positions
   - Use central difference: ∂A/∂B ≈ (A+ - A-)/(2δ)
   - Project onto C direction

### Validation

The implementation includes validation of:
1. Antisymmetry relations
2. Frenet-Serret formulas (κ and τ)
3. Vanishing of self-components

## Physical Interpretation

### Curvature κ = (∂T/∂T)·n
- Measures how rapidly the field line bends
- Units: 1/length
- Zero for straight field lines

### Torsion τ = (∂n/∂T)·b
- Measures the rotation of the osculating plane
- Units: 1/length
- Zero for planar curves (e.g., dipole field lines in meridional planes)

### Normal Derivatives
- Describe how the frame changes when moving perpendicular to the field line
- Important for understanding field line bundles and flux tubes

### Binormal Derivatives
- Describe how the frame changes when moving in the binormal direction
- Related to field line twisting and 3D structure

## Example Values for a Dipole Field

In the equatorial plane of a dipole field:
- κ = 2/r (analytical)
- τ = 0 (field lines are planar)
- (∂T/∂n)·n relates to the divergence of nearby field lines
- Other derivatives describe the geometric structure of the dipole

## Numerical Considerations

1. **Step size selection**: Default δ = 0.01 Re balances accuracy and numerical stability

2. **Boundary handling**: Care needed near model boundaries or singular points

3. **Accuracy**: Finite differences have O(δ²) error for smooth fields

4. **Performance**: Fully vectorized for efficient array processing

## References

1. do Carmo, M. P. (1976). Differential Geometry of Curves and Surfaces.
2. Frenet, F. (1852). Sur les courbes à double courbure.
3. Serret, J. A. (1851). Sur quelques formules relatives à la théorie des courbes à double courbure.