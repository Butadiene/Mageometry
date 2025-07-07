# Field Line Directional Derivatives Design

## Overview

This document outlines the design for implementing directional derivatives of magnetic field line geometry vectors (tangent, normal, and binormal) in the geopack-vectorize library.

## Current Implementation Status

### Existing Functions (Confirmed)
- **Curvature (κ)**: `field_line_curvature_vectorized()` 
  - Definition: κ = |dT/ds| (magnitude of tangential derivative of unit tangent)
  - Implementation: Uses finite differences along field line
  - Note: The normal component of dT/ds is κ, which is why κ = |dT/ds|
  
- **Torsion (τ)**: `field_line_torsion_vectorized()`
  - Definition: τ = -N·(dB/ds) (tangential derivative of binormal)
  - Implementation: Computes rate of rotation of osculating plane
  - Note: This is consistent with dN/ds having binormal component τ

## Mathematical Framework

### Directional Derivative Definition
For a vector field **V** and unit direction vector **u**, the directional derivative is:

```
∇_u V = (u · ∇)V = u_x ∂V/∂x + u_y ∂V/∂y + u_z ∂V/∂z
```

### Frenet-Serret Frame
- **T**: Unit tangent vector (along field line)
- **N**: Principal normal vector (direction of curvature)
- **B**: Binormal vector (B = T × N)

### Frenet-Serret Formulas and Compatibility Relations
The Frenet-Serret formulas describe how the frame vectors change along the curve:
- dT/ds = κN
- dN/ds = -κT + τB
- dB/ds = -τN

From orthonormality constraints (T·N = 0, etc.) and the Frenet-Serret formulas, we derive:
- (N·∇)N has tangent component = -κ (from T·N = 0 constraint)
- (B·∇)N has tangent component = -τ (from geometric compatibility)

### Antisymmetry and Self-Component Properties
From the orthonormality of the Frenet frame (T·T = N·N = B·B = 1, T·N = T·B = N·B = 0), important properties emerge:

1. **Self-component vanishes**: The component of (A·∇)A in the A direction is always zero
   - (T·∇)T has zero T-component
   - (N·∇)N has zero N-component
   - (B·∇)B has zero B-component

2. **Antisymmetry property**: If (A·∇)B has component c in the C direction, then (C·∇)B has component -c in the A direction
   - Example: If (N·∇)T has component κ in the N direction, then (N·∇)N has component -κ in the T direction
   - Example: If (B·∇)N has component -τ in the T direction, then (T·∇)N has component τ in the B direction

These properties provide powerful validation checks for our implementations.

### Required Derivatives

#### Tangential Derivatives (Frenet-Serret formulas)
These are already implemented in the existing code:

1. **Tangential derivative of tangent**: ∂T/∂s = (T · ∇)T = dT/ds
   - Components: tangent component = 0 (self-component property), normal component = κ (curvature), binormal component = 0
   - This is the definition of curvature: κ = |dT/ds|

2. **Tangential derivative of normal**: ∂N/∂s = (T · ∇)N = dN/ds
   - Components: tangent component = -κ (negative curvature), normal component = 0, binormal component = τ (torsion)
   - From Frenet-Serret: dN/ds = -κT + τB

3. **Tangential derivative of binormal**: ∂B/∂s = (T · ∇)B = dB/ds
   - Components: tangent component = 0, normal component = -τ (negative torsion), binormal component = 0
   - From Frenet-Serret: dB/ds = -τN

#### Normal Derivatives (New implementations needed)

1. **Normal derivative of tangent**: ∂T/∂n = (N · ∇)T
   - Components: tangent component = 0 (self-component property), normal and binormal components
   
2. **Normal derivative of normal**: ∂N/∂n = (N · ∇)N  
   - Components: tangent component = -κ (negative curvature), normal component = 0 (self-component property), binormal component
   
#### Binormal Derivatives (New implementations needed)

3. **Binormal derivative of tangent**: ∂T/∂b = (B · ∇)T
   - Components: tangent component = 0 (self-component property), normal and binormal components
   
4. **Binormal derivative of normal**: ∂N/∂b = (B · ∇)N
   - Components: tangent component = -τ (negative torsion), normal component = 0 (orthogonality), binormal component = 0 (self-component property)

## Proposed Implementation

### Function Signatures

```python
def field_line_tangent_normal_derivative_vectorized(
    model_func, parmod, ps, x, y, z, delta=0.01
):
    """
    Calculate the normal derivative of the tangent vector: ∂T/∂n = (N · ∇)T
    
    Parameters
    ----------
    model_func : callable
        Magnetic field model function
    parmod : array_like
        Model parameters
    ps : float
        Dipole tilt angle in radians
    x, y, z : float or array_like
        Position coordinates in GSM system (Re)
    delta : float, optional
        Step size for finite differences (Re), default 0.01
        
    Returns
    -------
    dT_dn_x, dT_dn_y, dT_dn_z : float or ndarray
        Cartesian components of ∂T/∂n
    dT_dn_normal : float or ndarray
        Normal component: (∂T/∂n)·N
    dT_dn_binormal : float or ndarray
        Binormal component: (∂T/∂n)·B
    """

def field_line_normal_normal_derivative_vectorized(
    model_func, parmod, ps, x, y, z, delta=0.01
):
    """
    Calculate the normal derivative of the normal vector: ∂N/∂n = (N · ∇)N
    
    Returns
    -------
    dN_dn_x, dN_dn_y, dN_dn_z : float or ndarray
        Cartesian components of ∂N/∂n
    dN_dn_tangent : float or ndarray
        Tangent component: (∂N/∂n)·T = -κ (negative curvature)
    dN_dn_binormal : float or ndarray
        Binormal component: (∂N/∂n)·B
    """

def field_line_tangent_binormal_derivative_vectorized(
    model_func, parmod, ps, x, y, z, delta=0.01
):
    """
    Calculate the binormal derivative of the tangent vector: ∂T/∂b = (B · ∇)T
    
    Returns
    -------
    dT_db_x, dT_db_y, dT_db_z : float or ndarray
        Cartesian components of ∂T/∂b
    dT_db_normal : float or ndarray
        Normal component: (∂T/∂b)·N
    dT_db_binormal : float or ndarray
        Binormal component: (∂T/∂b)·B
    """

def field_line_normal_binormal_derivative_vectorized(
    model_func, parmod, ps, x, y, z, delta=0.01
):
    """
    Calculate the binormal derivative of the normal vector: ∂N/∂b = (B · ∇)N
    
    Returns
    -------
    dN_db_x, dN_db_y, dN_db_z : float or ndarray
        Cartesian components of ∂N/∂b
    dN_db_tangent : float or ndarray
        Tangent component: (∂N/∂b)·T = -τ (negative torsion)
    dN_db_binormal : float or ndarray
        Binormal component: (∂N/∂b)·B
    """
```

### Implementation Algorithm

#### Step 1: Compute Gradient Tensor
For each vector field V = (Vx, Vy, Vz), compute the 3×3 gradient tensor:

```python
# Using finite differences with step size delta
def compute_gradient_tensor(compute_vector_func, x, y, z, delta):
    # Forward and backward steps in each direction
    V_xplus = compute_vector_func(x + delta, y, z)
    V_xminus = compute_vector_func(x - delta, y, z)
    V_yplus = compute_vector_func(x, y + delta, z)
    V_yminus = compute_vector_func(x, y - delta, z)
    V_zplus = compute_vector_func(x, y, z + delta)
    V_zminus = compute_vector_func(x, y, z - delta)
    
    # Central differences
    dVx_dx = (V_xplus[0] - V_xminus[0]) / (2 * delta)
    dVy_dx = (V_xplus[1] - V_xminus[1]) / (2 * delta)
    dVz_dx = (V_xplus[2] - V_xminus[2]) / (2 * delta)
    
    dVx_dy = (V_yplus[0] - V_yminus[0]) / (2 * delta)
    dVy_dy = (V_yplus[1] - V_yminus[1]) / (2 * delta)
    dVz_dy = (V_yplus[2] - V_yminus[2]) / (2 * delta)
    
    dVx_dz = (V_zplus[0] - V_zminus[0]) / (2 * delta)
    dVy_dz = (V_zplus[1] - V_zminus[1]) / (2 * delta)
    dVz_dz = (V_zplus[2] - V_zminus[2]) / (2 * delta)
    
    return [[dVx_dx, dVx_dy, dVx_dz],
            [dVy_dx, dVy_dy, dVy_dz],
            [dVz_dx, dVz_dy, dVz_dz]]
```

#### Step 2: Apply Directional Derivative
```python
def directional_derivative(grad_tensor, direction):
    # direction = (ux, uy, uz) is a unit vector
    dV_du_x = direction[0] * grad_tensor[0][0] + \
              direction[1] * grad_tensor[0][1] + \
              direction[2] * grad_tensor[0][2]
    
    dV_du_y = direction[0] * grad_tensor[1][0] + \
              direction[1] * grad_tensor[1][1] + \
              direction[2] * grad_tensor[1][2]
    
    dV_du_z = direction[0] * grad_tensor[2][0] + \
              direction[1] * grad_tensor[2][1] + \
              direction[2] * grad_tensor[2][2]
    
    return dV_du_x, dV_du_y, dV_du_z
```

#### Step 3: Project onto Frenet Frame
```python
def project_to_frenet_components(vector, T, N, B):
    # Components in Frenet frame
    tangent_component = vector[0] * T[0] + vector[1] * T[1] + vector[2] * T[2]
    normal_component = vector[0] * N[0] + vector[1] * N[1] + vector[2] * N[2]
    binormal_component = vector[0] * B[0] + vector[1] * B[1] + vector[2] * B[2]
    
    return tangent_component, normal_component, binormal_component
```

### Key Implementation Considerations

1. **Efficiency Optimizations**
   - Cache Frenet frame calculations (T, N, B) to avoid redundant computations
   - Use vectorized numpy operations for batch processing
   - Consider reusing gradient calculations where possible

2. **Numerical Stability**
   - Use appropriate step size (default δ = 0.01 Re)
   - Handle edge cases (zero field regions, straight field lines)
   - Implement bounds checking for model validity regions

3. **Validation Constraints**
   - Verify orthogonality: T·N = 0, T·B = 0, N·B = 0
   - Check derivative constraints: ∂(T·N)/∂u = 0 for any direction u
   - Verify Frenet-Serret relationships:
     - (∂N/∂n)·T = -κ (negative curvature)
     - (∂N/∂b)·T = -τ (negative torsion)
   - Verify self-component properties:
     - (∂T/∂t)·T = 0 (tangent component of tangent derivative in tangent direction)
     - (∂N/∂n)·N = 0 (normal component of normal derivative in normal direction)
     - (∂B/∂b)·B = 0 (binormal component of binormal derivative in binormal direction)
   - Verify antisymmetry relations:
     - (∂T/∂n)·N = κ and (∂N/∂n)·T = -κ
     - (∂N/∂b)·T = -τ and (∂T/∂b)·N = τ
     - (∂T/∂n)·B and (∂B/∂n)·T should be negatives of each other
   - Verify Frenet-Serret tangential derivatives:
     - (∂T/∂s)·N = κ (curvature definition)
     - (∂N/∂s)·T = -κ and (∂N/∂s)·B = τ
     - (∂B/∂s)·N = -τ
   - Validate against analytical solutions for simple field configurations

4. **Error Handling**
   - Check for NaN/inf values in low field regions
   - Ensure proper scalar/array compatibility
   - Handle boundary conditions appropriately

## Testing Strategy

1. **Unit Tests**
   - Test scalar and array inputs
   - Verify orthogonality constraints are preserved
   - Check antisymmetry properties:
     ```python
     # Example test for antisymmetry
     dT_dn_N = (∂T/∂n)·N  # Should equal κ
     dN_dn_T = (∂N/∂n)·T  # Should equal -κ
     assert np.allclose(dT_dn_N + dN_dn_T, 0)
     ```
   - Verify self-component vanishing:
     ```python
     # Example test for self-component
     dN_dn_N = (∂N/∂n)·N  # Should be zero
     assert np.allclose(dN_dn_N, 0)
     ```

2. **Validation Tests**
   - Compare with analytical solutions for dipole field
   - Verify Frenet-Serret relations
   - Test limiting cases (straight field lines, circular loops)

3. **Performance Tests**
   - Benchmark against scalar implementations
   - Measure speedup for various array sizes
   - Profile memory usage

## Integration Plan

1. Add functions to `geopack/vectorized/field_line_geometry_vectorized.py`
2. Export functions in `geopack/__init__.py`
3. Create comprehensive test suite in `tests/test_field_line_directional_derivatives.py`
4. Add example notebook demonstrating usage
5. Update documentation with mathematical details and usage examples

## Future Extensions

1. **Higher-Order Derivatives**
   - Second directional derivatives
   - Mixed partial derivatives

2. **Additional Geometric Properties**
   - Geodesic curvature
   - Normal curvature in specific directions

3. **Optimization**
   - GPU acceleration for large-scale calculations
   - Adaptive step size selection
   - Parallel processing for independent positions