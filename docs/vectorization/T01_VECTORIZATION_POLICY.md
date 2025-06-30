# T01 Vectorization Implementation Policy

## Overview

This document outlines the comprehensive policy for vectorizing the T01 magnetospheric field model. The T01 model presents unique challenges due to its iterative algorithms, complex conditional logic, and multi-region field calculations. This policy builds upon the successful T96 vectorization approach while addressing T01-specific complexities.

## Core Challenges and Solutions

### 1. Iterative Algorithms

#### 1.1 Sigma Calculation Iterator
The T01 model contains an iterative algorithm to find unwarped coordinates:

```python
# Scalar version (current)
dd = 1.
while dd > 1e-6:
    xsold = xss
    zsold = zss
    rh = rh0 + rh2 * (zss/r)**2
    sinpsas = sps / (1 + (r/rh)**3)**0.33333333
    cospsas = np.sqrt(1 - sinpsas**2)
    zss = x * sinpsas + z * cospsas
    xss = x * cospsas - z * sinpsas
    dd = np.abs(xss - xsold) + np.abs(zss - zsold)
```

**Vectorized Solution:**
```python
# Vectorized version - Option 1: Masked updates
def iterate_sigma_vectorized_masked(x, y, z, sps, rh0, rh2, max_iter=50):
    """Vectorized iterative sigma calculation with masked updates."""
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    r = np.sqrt(x**2 + y**2 + z**2)
    xss = x.copy()
    zss = z.copy()
    
    # Track convergence for each point
    converged = np.zeros_like(x, dtype=bool)
    
    for i in range(max_iter):
        xsold = xss.copy()
        zsold = zss.copy()
        
        # Only update non-converged points
        active = ~converged
        if not np.any(active):
            break
            
        rh = rh0 + rh2 * (zss[active]/r[active])**2
        sinpsas = sps / (1 + (r[active]/rh)**3)**0.33333333
        cospsas = np.sqrt(1 - sinpsas**2)
        
        zss[active] = x[active] * sinpsas + z[active] * cospsas
        xss[active] = x[active] * cospsas - z[active] * sinpsas
        
        # Check convergence
        dd = np.abs(xss - xsold) + np.abs(zss - zsold)
        converged = dd < 1e-6
    
    return xss, zss

# Vectorized version - Option 2: Full array operations (potentially more efficient)
def iterate_sigma_vectorized_full(x, y, z, sps, rh0, rh2, max_iter=50):
    """Vectorized iterative sigma calculation with full array operations."""
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    r = np.sqrt(x**2 + y**2 + z**2)
    xss = x.copy()
    zss = z.copy()
    
    # Track convergence for each point
    converged = np.zeros_like(x, dtype=bool)
    
    for i in range(max_iter):
        xsold = xss.copy()
        zsold = zss.copy()
        
        # Calculate updates for ALL points (leverages NumPy's optimized loops)
        rh = rh0 + rh2 * (zsold/r)**2
        sinpsas = sps / np.power(1 + (r/rh)**3, 1/3)
        sinpsas = np.clip(sinpsas, -1.0, 1.0)  # Ensure valid range
        cospsas = np.sqrt(np.maximum(1 - sinpsas**2, 0))
        
        new_xss = x * cospsas - z * sinpsas
        new_zss = x * sinpsas + z * cospsas
        
        # Conditionally apply updates only to non-converged points
        xss = np.where(converged, xsold, new_xss)
        zss = np.where(converged, zsold, new_zss)
        
        # Update convergence mask
        dd = np.abs(xss - xsold) + np.abs(zss - zsold)
        newly_converged = dd < 1e-6
        converged = np.logical_or(converged, newly_converged)
        
        if np.all(converged):
            break
    
    return xss, zss
```

#### 1.2 Performance Optimization for Iterations
- Use fixed maximum iterations with early exit when all points converge
- Process only non-converged points in each iteration
- Pre-allocate arrays to avoid repeated memory allocation
- Consider adaptive tolerance based on distance from origin

### 2. Complex Conditional Logic

#### 2.1 Three-Region Magnetosphere Model
The T01 model divides space into three regions based on sigma parameter:

```python
# Vectorized region processing
def t01_vectorized(parmod, ps, x, y, z):
    # Calculate sigma for all points
    sigma = calculate_sigma_vectorized(x, y, z, am, x0, rh0, rh2, sps)
    
    # Define region masks
    mask_inside = sigma < (s0 - dsig)
    mask_layer = (sigma >= (s0 - dsig)) & (sigma < (s0 + dsig))
    mask_outside = sigma >= (s0 + dsig)
    
    # Initialize output arrays
    bx = np.zeros_like(x)
    by = np.zeros_like(y)
    bz = np.zeros_like(z)
    
    # Process each region with appropriate physics
    if np.any(mask_inside):
        idx = mask_inside
        bx[idx], by[idx], bz[idx] = calculate_internal_field_vectorized(
            x[idx], y[idx], z[idx], parmod, ps
        )
    
    if np.any(mask_layer):
        idx = mask_layer
        # Interpolate between internal and external fields
        bx[idx], by[idx], bz[idx] = interpolate_boundary_field_vectorized(
            x[idx], y[idx], z[idx], sigma[idx], s0, dsig, parmod, ps
        )
    
    if np.any(mask_outside):
        idx = mask_outside
        # External field minus dipole
        bx[idx], by[idx], bz[idx] = calculate_external_field_vectorized(
            x[idx], y[idx], z[idx], parmod, ps
        )
```

#### 2.2 Component Selection Logic
T01 allows selective calculation of field components through option flags:

```python
def extall_vectorized(iopgen, iopt, iopb, iopr, a, ntot, pdyn, dst, 
                      byimf, bzimf, vbimf1, vbimf2, ps, x, y, z):
    """Vectorized external field calculation with component selection."""
    
    # Initialize all components
    bxcf = bycf = bzcf = np.zeros_like(x)
    bxt1 = byt1 = bzt1 = np.zeros_like(x)
    bxt2 = byt2 = bzt2 = np.zeros_like(x)
    
    # Calculate components based on flags
    if iopgen <= 1:  # Dipole shielding
        bxcf, bycf, bzcf = shlcar3x3_vectorized(x, y, z, ps)
        bxcf *= xappa3
        bycf *= xappa3
        bzcf *= xappa3
    
    if (iopgen == 0) or (iopgen == 2):  # Tail field
        if iopt == 0 or iopt == 1:  # Mode 1
            bxt1, byt1, bzt1 = tail_mode1_vectorized(x, y, z, ps)
        if iopt == 0 or iopt == 2:  # Mode 2
            bxt2, byt2, bzt2 = tail_mode2_vectorized(x, y, z, ps)
    
    # Continue for other components...
```

### 3. Global State Management

#### 3.1 Parameter Propagation
T01 uses global variables for model parameters. Vectorized version should eliminate globals:

```python
@dataclass
class T01Parameters:
    """Container for T01 model parameters.
    
    All fields can be either scalars or arrays of the same shape as the input coordinates.
    This allows for different solar wind conditions at each point.
    """
    dxshift1: Union[float, np.ndarray]
    dxshift2: Union[float, np.ndarray]
    d: Union[float, np.ndarray]
    deltady: Union[float, np.ndarray]
    xkappa1: Union[float, np.ndarray]
    xkappa2: Union[float, np.ndarray]
    sc_sy: Union[float, np.ndarray]
    sc_pr: Union[float, np.ndarray]
    phi: Union[float, np.ndarray]
    g: Union[float, np.ndarray]
    rh0: Union[float, np.ndarray]
    xappa: Union[float, np.ndarray]  # Pressure scaling factor
    
def calculate_parameters(parmod, ps, a, n_points=None):
    """Calculate all T01 parameters from input.
    
    Parameters
    ----------
    parmod : array_like
        Model parameters, shape (6,) or (n_points, 6)
        [pdyn, dst, byimf, bzimf, g1, g2]
    ps : float
        Dipole tilt angle
    a : array_like
        Model coefficients array
    n_points : int, optional
        Number of points (used when parmod is 1D)
        
    Returns
    -------
    params : T01Parameters
        Container with all calculated parameters
    """
    # Handle both scalar and array inputs
    parmod = np.atleast_2d(parmod)
    if parmod.shape[0] == 1 and n_points is not None:
        # Broadcast scalar parameters to all points
        parmod = np.repeat(parmod, n_points, axis=0)
    
    # Extract parameters (now potentially arrays)
    pdyn = parmod[:, 0]
    dst = parmod[:, 1]
    byimf = parmod[:, 2]
    bzimf = parmod[:, 3]
    g1 = parmod[:, 4]
    g2 = parmod[:, 5]
    
    # Pressure scaling
    xappa = (pdyn / 2.0) ** a[38]
    
    # Initialize parameters container
    params = T01Parameters(
        dxshift1=a[25] + a[26] * g2,
        dxshift2=np.zeros_like(pdyn),
        d=np.full_like(pdyn, a[27]),
        deltady=np.full_like(pdyn, a[28]),
        xkappa1=a[34] + a[35] * g2,
        xkappa2=a[36] + a[37] * g2,
        sc_sy=None,  # Calculated below
        sc_pr=None,  # Calculated below
        phi=None,    # Calculated below
        g=np.full_like(pdyn, a[40]),
        rh0=np.full_like(pdyn, a[39]),
        xappa=xappa
    )
    
    # Ring current parameters (vectorized)
    params.phi = 1.5707963 * np.tanh(np.abs(dst) / a[33])
    znam = np.maximum(np.abs(dst), 20.0)
    params.sc_sy = a[29] * (20/znam)**a[30] * xappa
    params.sc_pr = a[31] * (20/znam)**a[32] * xappa
    
    # If single point, convert arrays back to scalars
    if parmod.shape[0] == 1:
        for field in params.__dataclass_fields__:
            value = getattr(params, field)
            if isinstance(value, np.ndarray) and value.size == 1:
                setattr(params, field, value.item())
    
    return params
```

### 4. Vectorization Patterns

#### 4.1 Safe Mathematical Operations
Following T96 patterns with T01-specific adaptations:

```python
# Safe square root for cospsas calculation
sinpsas = sps / np.power(1 + (r/rh)**3, 1/3)
sinpsas = np.clip(sinpsas, -1.0, 1.0)  # Ensure valid range
cospsas = np.sqrt(np.maximum(1 - sinpsas**2, 0))

# Safe division in boundary calculations
xmxm = am + xss - x0
xmxm = np.maximum(xmxm, 0)  # Cylinder boundary condition

# Safe discriminant for sigma
discriminant = (aro + axx0)**2 - 4.0 * asq * axx0
discriminant = np.maximum(discriminant, 0)
sigma = np.sqrt((aro + axx0 + np.sqrt(discriminant)) / (2.0 * asq))
```

#### 4.2 Efficient Harmonic Calculations
Vectorize the shlcar3x3 18-harmonic expansion:

```python
def shlcar3x3_vectorized_partial(x, y, z, ps, a):
    """Vectorized 3x3x2 Cartesian harmonic shield with loop optimization."""
    # Pre-calculate all tilted coordinates
    x1 = x * ct1 - z * st1
    z1 = x * st1 + z * ct1
    x2 = x * ct2 - z * st2
    z2 = x * st2 + z * ct2
    
    # Initialize output arrays
    bx = np.zeros_like(x)
    by = np.zeros_like(y)
    bz = np.zeros_like(z)
    
    # Vectorize harmonic calculations
    # Group by p-values to minimize redundant calculations
    for p_idx, p in enumerate([p1, p2, p3]):
        cyp = np.cos(y / p)
        syp = np.sin(y / p)
        
        for r_idx, r in enumerate([r1, r2, r3]):
            sqpr = np.sqrt(1/p**2 + 1/r**2)
            czr = np.cos(z1 / r)
            szr = np.sin(z1 / r)
            expr = np.exp(sqpr * x1)
            
            # Calculate harmonics
            fx = -sqpr * expr * cyp * szr
            hy = expr / p * syp * szr
            fz = -expr * cyp / r * czr
            
            # Apply coefficients and accumulate
            coeff_idx = p_idx * 6 + r_idx * 2
            bx += a[coeff_idx] * (fx * ct1 + fz * st1)
            by += a[coeff_idx] * hy
            bz += a[coeff_idx] * (-fx * st1 + fz * ct1)
    
    return bx, by, bz

def shlcar3x3_fully_vectorized(x, y, z, ps, a):
    """Fully vectorized 3x3x2 Cartesian harmonic shield - no loops."""
    # Extract parameters and reshape for broadcasting
    p_vals = a[36:39].reshape(3, 1)  # Shape (3, 1) for p1, p2, p3
    r_vals = a[39:42].reshape(1, 3)  # Shape (1, 3) for r1, r2, r3
    
    # Tilt angles
    t1, t2 = a[48], a[49]
    ct1, st1 = np.cos(ps * t1), np.sin(ps * t1)
    ct2, st2 = np.cos(ps * t2), np.sin(ps * t2)
    
    # Pre-calculate tilted coordinates
    x1 = x * ct1 - z * st1
    z1 = x * st1 + z * ct1
    x2 = x * ct2 - z * st2
    z2 = x * st2 + z * ct2
    
    # Reshape inputs for broadcasting with (3, 3, N) shape
    x_bc = x.reshape(1, 1, -1)
    y_bc = y.reshape(1, 1, -1)
    z1_bc = z1.reshape(1, 1, -1)
    x1_bc = x1.reshape(1, 1, -1)
    
    # Calculate all harmonics at once using broadcasting
    # Shape: (3, 1, 1) * (1, 1, N) = (3, 1, N)
    cyp = np.cos(y_bc / p_vals.reshape(3, 1, 1))
    syp = np.sin(y_bc / p_vals.reshape(3, 1, 1))
    
    # Shape: (1, 3, 1) * (1, 1, N) = (1, 3, N)
    czr = np.cos(z1_bc / r_vals.reshape(1, 3, 1))
    szr = np.sin(z1_bc / r_vals.reshape(1, 3, 1))
    
    # Shape: (3, 3, 1)
    sqpr = np.sqrt(1/p_vals**2 + 1/r_vals**2).reshape(3, 3, 1)
    
    # Shape: (3, 3, N)
    expr = np.exp(sqpr * x1_bc)
    
    # Calculate field components - Shape: (3, 3, N)
    fx = -sqpr * expr * cyp * szr
    hy = expr / p_vals.reshape(3, 1, 1) * syp * szr
    fz = -expr * cyp / r_vals.reshape(1, 3, 1) * czr
    
    # Transform and apply coefficients
    hx = fx * ct1 + fz * st1
    hz = -fx * st1 + fz * ct1
    
    # Reshape coefficients for broadcasting
    coeff = a[:18].reshape(3, 3, 2)  # Shape (3, 3, 2) for 18 harmonics
    
    # Apply coefficients and sum - using Einstein summation for clarity
    bx = np.einsum('ijk,ijk...->...', coeff[..., 0], hx)
    by = np.einsum('ijk,ijk...->...', coeff[..., 0], hy)
    bz = np.einsum('ijk,ijk...->...', coeff[..., 0], hz)
    
    # Add contributions from second tilt angle (similar process for x2, z2)
    # ... (implementation details for second set of harmonics)
    
    return bx, by, bz
```

### 5. Testing and Validation Strategy

#### 5.1 Comprehensive Test Coverage

##### Spatial Regions
- **Near-Earth**: 1.5-3 Re (dense sampling)
- **Mid-magnetosphere**: 3-10 Re
- **Valid tail region**: -15 to -1 Re (model limit)
- **Boundary layer**: Points near sigma = s0 ± dsig
- **High-latitude**: Large z values

##### Parameter Space
```python
# T01-specific parameters
pdyn: 0.5-10.0 nPa
dst: -200 to +50 nT
byimf: -10 to +10 nT
bzimf: -10 to +10 nT
g1: -10 to +10 (storm-time index)
g2: -10 to +10 (storm-time index)
ps: -0.5 to +0.5 radians
```

##### Edge Cases
- Points at x = -15 Re (model boundary)
- Zero IMF conditions
- Extreme storm conditions (large |dst|)
- Points on coordinate axes
- Iterative algorithm convergence limits

#### 5.2 Validation Metrics

##### Accuracy Requirements
```python
def validate_accuracy(scalar_func, vector_func, test_points, params):
    """Validate vectorized implementation accuracy."""
    max_rel_error = 0.0
    convergence_failures = 0
    
    for i, (x, y, z) in enumerate(test_points):
        # Scalar calculation
        bx_s, by_s, bz_s = scalar_func(params, ps, x, y, z)
        
        # Vector calculation
        bx_v, by_v, bz_v = vector_func(params, ps, x, y, z)
        
        # Relative error
        b_mag = np.sqrt(bx_s**2 + by_s**2 + bz_s**2)
        if b_mag > 1e-3:  # Avoid tiny field regions
            diff_mag = np.sqrt((bx_v-bx_s)**2 + (by_v-by_s)**2 + (bz_v-bz_s)**2)
            rel_error = diff_mag / b_mag
            max_rel_error = max(max_rel_error, rel_error)
        
        # Check iterative convergence
        if not check_convergence(x, y, z):
            convergence_failures += 1
    
    assert max_rel_error < 1e-6, f"Max relative error {max_rel_error} exceeds threshold"
    assert convergence_failures == 0, f"{convergence_failures} points failed to converge"
```

##### Performance Targets
- Single point overhead: < 2x scalar version
- Batch processing (1000 points): > 20x speedup
- Memory usage: O(n) with minimal temporary arrays
- Iteration efficiency: < 20 iterations average

#### 5.3 Component Testing

Test each component independently before integration:

1. **Iterative sigma calculation**
   - Convergence for all test points
   - Accuracy vs scalar iteration
   - Performance with early exit optimization

2. **Harmonic expansions (shlcar3x3)**
   - Each of 18 harmonics individually
   - Combined field accuracy
   - Tilt angle handling

3. **Field components**
   - Dipole shielding
   - Tail fields (modes 1 & 2)
   - Birkeland currents (4 terms)
   - Ring currents (SRC & PRC)
   - Interconnection field

4. **Region interpolation**
   - Smooth transitions at boundaries
   - Correct blending factors
   - Conservation of field divergence

### 6. Implementation Guidelines

#### 6.1 Function Structure Template

```python
def component_t01_vectorized(x, y, z, params: T01Parameters):
    """Vectorized T01 component calculation.
    
    Parameters
    ----------
    x, y, z : array_like
        GSM coordinates in Re
    params : T01Parameters
        Model parameters container
        
    Returns
    -------
    bx, by, bz : ndarray
        Magnetic field components in nT
    """
    # Input validation and conversion
    scalar_input = np.isscalar(x) and np.isscalar(y) and np.isscalar(z)
    x = np.atleast_1d(x)
    y = np.atleast_1d(y) 
    z = np.atleast_1d(z)
    x, y, z = np.broadcast_arrays(x, y, z)
    
    # Initialize output arrays
    bx = np.zeros_like(x)
    by = np.zeros_like(y)
    bz = np.zeros_like(z)
    
    # Vectorized calculations
    # ...
    
    # Preserve input shape
    if scalar_input:
        return bx.item(), by.item(), bz.item()
    return bx, by, bz
```

#### 6.2 Code Organization

```
geopack/
├── t01_vectorized.py          # Main vectorized T01 implementation
├── t01_components/            # Component implementations
│   ├── __init__.py
│   ├── iterative_sigma.py    # Iterative algorithm
│   ├── shlcar3x3_vec.py      # Harmonic shield  
│   ├── tail_modes_vec.py     # Tail field modes
│   ├── birkeland_vec.py      # Birkeland currents
│   ├── ring_current_vec.py   # Ring current
│   └── interconnect_vec.py   # Interconnection field
└── t01_utils.py              # Shared utilities and parameters
```

#### 6.3 Progressive Implementation Plan

1. **Phase 1: Core Infrastructure**
   - Parameter container and management
   - Iterative sigma algorithm
   - Basic input/output handling

2. **Phase 2: Simple Components**
   - Dipole and shielding (shlcar3x3)
   - Interconnection field
   - Component integration framework

3. **Phase 3: Complex Components**
   - Tail field modes with warping
   - Birkeland current system
   - Ring current with asymmetries

4. **Phase 4: Integration and Optimization**
   - Full model assembly
   - Region interpolation
   - Performance optimization
   - Comprehensive validation

### 7. Special Considerations for T01

#### 7.1 Model Validity Limits
```python
def validate_input_range(x, y, z):
    """Check if coordinates are within T01 validity range."""
    # T01 is only valid for x > -15 Re
    invalid_mask = x < -15.0
    if np.any(invalid_mask):
        warnings.warn(
            f"T01 model is only valid for x > -15 Re. "
            f"Found {np.sum(invalid_mask)} points outside valid range."
        )
    return ~invalid_mask
```

#### 7.2 Storm-Time Index Handling
```python
def process_storm_indices(g1, g2, n_points):
    """Handle G1 and G2 storm-time indices."""
    # Ensure arrays for vectorized operations
    if np.isscalar(g1):
        g1 = np.full(n_points, g1)
    if np.isscalar(g2):
        g2 = np.full(n_points, g2)
    
    # Validate ranges (typical: -10 to +10)
    g1 = np.clip(g1, -10.0, 10.0)
    g2 = np.clip(g2, -10.0, 10.0)
    
    return g1, g2
```

#### 7.3 Iterative Algorithm Monitoring
```python
class IterationMonitor:
    """Monitor iterative algorithm performance."""
    
    def __init__(self):
        self.iteration_counts = []
        self.convergence_times = []
        
    def record(self, n_iterations, elapsed_time, n_points):
        self.iteration_counts.append(n_iterations)
        self.convergence_times.append(elapsed_time)
        
    def report(self):
        return {
            'mean_iterations': np.mean(self.iteration_counts),
            'max_iterations': np.max(self.iteration_counts),
            'mean_time_per_point': np.mean(self.convergence_times),
        }
```

### 8. Performance Optimization Strategies

#### 8.1 Advanced Iteration Optimization

##### 8.1.1 Benchmarking Both Iteration Patterns
```python
def benchmark_iteration_patterns(x, y, z, sps, rh0, rh2):
    """Compare performance of masked vs full array iteration."""
    import time
    
    # Pattern 1: Masked updates
    start = time.perf_counter()
    xss1, zss1 = iterate_sigma_vectorized_masked(x, y, z, sps, rh0, rh2)
    time_masked = time.perf_counter() - start
    
    # Pattern 2: Full array operations
    start = time.perf_counter()
    xss2, zss2 = iterate_sigma_vectorized_full(x, y, z, sps, rh0, rh2)
    time_full = time.perf_counter() - start
    
    # Verify results match
    assert np.allclose(xss1, xss2, rtol=1e-10)
    assert np.allclose(zss1, zss2, rtol=1e-10)
    
    return {
        'masked_time': time_masked,
        'full_time': time_full,
        'speedup': time_masked / time_full
    }
```

##### 8.1.2 Adaptive Convergence Strategy
```python
def iterate_sigma_adaptive(x, y, z, sps, rh0, rh2, max_iter=50):
    """Adaptive convergence with variable tolerance based on location."""
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Adaptive tolerance: tighter near Earth, looser in far tail
    base_tol = 1e-6
    tolerance = base_tol * (1 + 0.1 * np.maximum(np.abs(x) - 10, 0))
    
    # Initialize
    xss = x.copy()
    zss = z.copy()
    converged = np.zeros_like(x, dtype=bool)
    iteration_count = np.zeros_like(x, dtype=int)
    
    for i in range(max_iter):
        # Track iterations per point for performance analysis
        iteration_count[~converged] += 1
        
        # ... (iteration logic) ...
        
        # Check convergence with adaptive tolerance
        dd = np.abs(xss - xsold) + np.abs(zss - zsold)
        newly_converged = dd < tolerance
        converged = np.logical_or(converged, newly_converged)
        
        if np.all(converged):
            break
    
    return xss, zss, iteration_count
```

##### 8.1.3 Compiled Iteration Core (Optional)
```python
from numba import njit, prange

@njit(parallel=True)
def iterate_sigma_compiled(x, y, z, sps, rh0, rh2, max_iter=50):
    """Numba-compiled iteration for maximum performance."""
    n = len(x)
    xss = x.copy()
    zss = z.copy()
    
    for idx in prange(n):
        xi, yi, zi = x[idx], y[idx], z[idx]
        r = np.sqrt(xi**2 + yi**2 + zi**2)
        xss_i = xi
        zss_i = zi
        
        for _ in range(max_iter):
            xsold = xss_i
            zsold = zss_i
            
            rh = rh0 + rh2 * (zss_i/r)**2
            sinpsas = sps / (1 + (r/rh)**3)**(1/3)
            cospsas = np.sqrt(1 - sinpsas**2)
            
            xss_i = xi * cospsas - zi * sinpsas
            zss_i = xi * sinpsas + zi * cospsas
            
            if abs(xss_i - xsold) + abs(zss_i - zsold) < 1e-6:
                break
        
        xss[idx] = xss_i
        zss[idx] = zss_i
    
    return xss, zss
```

#### 8.2 Memory Optimization

##### 8.2.1 Memory-Efficient Processing for Large Arrays
```python
def t01_vectorized_chunked(parmod, ps, x, y, z, chunk_size=10000):
    """Process large arrays in chunks to optimize memory usage."""
    n_points = len(x)
    
    if n_points <= chunk_size:
        # Process normally for small arrays
        return t01_vectorized(parmod, ps, x, y, z)
    
    # Initialize output arrays
    bx = np.empty_like(x)
    by = np.empty_like(y)
    bz = np.empty_like(z)
    
    # Process in chunks
    for start in range(0, n_points, chunk_size):
        end = min(start + chunk_size, n_points)
        
        # Extract chunk
        x_chunk = x[start:end]
        y_chunk = y[start:end]
        z_chunk = z[start:end]
        
        # Process chunk
        bx[start:end], by[start:end], bz[start:end] = \
            t01_vectorized(parmod, ps, x_chunk, y_chunk, z_chunk)
    
    return bx, by, bz
```

##### 8.2.2 In-Place Operations for Memory Efficiency
```python
def apply_warping_inplace(x, y, z, warp_params):
    """Apply coordinate warping in-place to save memory."""
    # Instead of creating new arrays, modify existing ones
    z -= 20 * warp_params.spsc
    z += warp_params.warp
    
    # For operations that need original values
    temp = x.copy()  # Only one temporary array
    x *= warp_params.cos_factor
    x -= z * warp_params.sin_factor
    z *= warp_params.cos_factor
    z += temp * warp_params.sin_factor
```

#### 8.3 Computational Optimization

##### 8.3.1 Pre-computation Strategy
```python
class T01PrecomputedData:
    """Pre-compute expensive operations for multiple evaluations."""
    
    def __init__(self, ps, a):
        # Pre-compute all tilt-dependent values
        self.sps = np.sin(ps)
        self.cps = np.cos(ps)
        
        # Pre-compute harmonic parameters
        self.p_vals = a[36:39]
        self.r_vals = a[39:42]
        self.t1, self.t2 = a[48:50]
        
        # Pre-compute tilt transforms
        self.ct1 = np.cos(ps * self.t1)
        self.st1 = np.sin(ps * self.t1)
        self.ct2 = np.cos(ps * self.t2)
        self.st2 = np.sin(ps * self.t2)
        
        # Pre-compute scale combinations
        self.sqpr_matrix = np.sqrt(1/self.p_vals[:, None]**2 + 
                                   1/self.r_vals[None, :]**2)

def t01_with_precompute(parmod, ps, x, y, z, precomputed=None):
    """Use pre-computed data for better performance."""
    if precomputed is None:
        precomputed = T01PrecomputedData(ps, a)
    
    # Use pre-computed values throughout calculation
    # ...
```

##### 8.3.2 SIMD-Friendly Array Layout
```python
def optimize_array_layout(x, y, z):
    """Ensure arrays are SIMD-friendly for vectorized operations."""
    # Ensure C-contiguous layout for better cache performance
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    z = np.ascontiguousarray(z)
    
    # Align arrays for SIMD operations (optional, platform-specific)
    # This ensures arrays start at addresses divisible by 32/64 bytes
    if x.dtype == np.float64:
        alignment = 64  # AVX-512 alignment
    else:
        alignment = 32  # AVX2 alignment
    
    # Create aligned arrays if needed
    if x.ctypes.data % alignment != 0:
        x_aligned = np.empty(x.shape, dtype=x.dtype, order='C')
        x_aligned[:] = x
        x = x_aligned
    
    return x, y, z
```

### 9. Documentation Requirements

#### 9.1 Function Documentation
Each vectorized function must include:
- Clear description of vectorization approach
- Input/output shape specifications
- Performance characteristics
- Accuracy guarantees
- Known limitations

#### 9.2 Testing Documentation
- Test coverage reports
- Accuracy validation results
- Performance benchmarks
- Comparison with scalar implementation

#### 9.3 User Guide
- Migration guide from scalar to vectorized
- Performance tuning recommendations
- Common usage patterns
- Troubleshooting guide

## 10. Special Considerations for T01-Specific Challenges

### 10.1 Handling Model Validity Warnings

```python
def t01_vectorized_with_validation(parmod, ps, x, y, z):
    """T01 with proper handling of validity warnings."""
    # Check validity range
    invalid_mask = x < -15.0
    
    # Initialize results
    bx = np.zeros_like(x)
    by = np.zeros_like(y)
    bz = np.zeros_like(z)
    
    # Only process valid points
    valid_mask = ~invalid_mask
    if np.any(valid_mask):
        bx[valid_mask], by[valid_mask], bz[valid_mask] = \
            t01_core_vectorized(parmod, ps, x[valid_mask], 
                                y[valid_mask], z[valid_mask])
    
    # Set invalid points to NaN or warning values
    if np.any(invalid_mask):
        warnings.warn(
            f"T01 model used outside valid range (x < -15 Re) "
            f"for {np.sum(invalid_mask)} points. "
            f"Results set to NaN for these points."
        )
        bx[invalid_mask] = np.nan
        by[invalid_mask] = np.nan
        bz[invalid_mask] = np.nan
    
    return bx, by, bz
```

### 10.2 Benchmarking Suite for T01

```python
def create_t01_benchmark_suite():
    """Comprehensive benchmark suite for T01 vectorization."""
    
    benchmarks = {
        'iteration_convergence': {
            'near_earth': generate_points(r_min=1.5, r_max=3, n=1000),
            'mid_distance': generate_points(r_min=5, r_max=10, n=1000),
            'tail_region': generate_tail_points(x_min=-15, x_max=-5, n=1000),
        },
        'storm_conditions': {
            'quiet': {'dst': -10, 'g1': 0, 'g2': 0},
            'moderate': {'dst': -50, 'g1': 5, 'g2': 5},
            'storm': {'dst': -200, 'g1': 10, 'g2': 10},
        },
        'array_sizes': [1, 10, 100, 1000, 10000, 100000],
    }
    
    return benchmarks

def run_t01_benchmarks(scalar_func, vector_func):
    """Run comprehensive benchmarks comparing implementations."""
    results = {}
    suite = create_t01_benchmark_suite()
    
    # Test iteration convergence by region
    for region, points in suite['iteration_convergence'].items():
        x, y, z = points
        _, _, iter_counts = iterate_sigma_adaptive(x, y, z, ps, rh0, rh2)
        results[f'iterations_{region}'] = {
            'mean': np.mean(iter_counts),
            'max': np.max(iter_counts),
            'convergence_rate': np.sum(iter_counts < 20) / len(iter_counts)
        }
    
    # Test performance vs array size
    for n in suite['array_sizes']:
        x = np.random.uniform(-10, 10, n)
        y = np.random.uniform(-10, 10, n)
        z = np.random.uniform(-5, 5, n)
        
        # Time scalar (loop)
        if n <= 100:  # Only for small arrays
            t_scalar = timeit.timeit(
                lambda: [scalar_func(parmod, ps, x[i], y[i], z[i]) 
                         for i in range(n)],
                number=10
            ) / 10
        else:
            t_scalar = np.nan
        
        # Time vectorized
        t_vector = timeit.timeit(
            lambda: vector_func(parmod, ps, x, y, z),
            number=10
        ) / 10
        
        results[f'size_{n}'] = {
            'scalar_time': t_scalar,
            'vector_time': t_vector,
            'speedup': t_scalar / t_vector if not np.isnan(t_scalar) else None,
            'points_per_second': n / t_vector
        }
    
    return results
```

### 10.3 Integration Testing Framework

```python
def test_t01_component_integration():
    """Test that all T01 components work together correctly."""
    
    # Test points covering all regions
    test_cases = [
        # Inside magnetosphere
        {'x': 5.0, 'y': 0.0, 'z': 0.0, 'region': 'inside'},
        # Boundary layer
        {'x': 8.0, 'y': 5.0, 'z': 0.0, 'region': 'boundary'},
        # Outside magnetosphere
        {'x': 15.0, 'y': 0.0, 'z': 0.0, 'region': 'outside'},
        # High latitude
        {'x': 3.0, 'y': 0.0, 'z': 5.0, 'region': 'high_lat'},
        # Tail region
        {'x': -10.0, 'y': 0.0, 'z': 0.0, 'region': 'tail'},
    ]
    
    # Test with different option flags
    option_combinations = [
        {'iopgen': 0, 'desc': 'all_components'},
        {'iopgen': 1, 'desc': 'dipole_shield_only'},
        {'iopgen': 2, 'desc': 'tail_only'},
        {'iopgen': 3, 'desc': 'birkeland_only'},
        {'iopgen': 4, 'desc': 'ring_current_only'},
        {'iopgen': 5, 'desc': 'interconnection_only'},
    ]
    
    for test_case in test_cases:
        for options in option_combinations:
            # Run scalar version
            bx_s, by_s, bz_s = extall_scalar(
                options['iopgen'], 0, 0, 0, a, 43,
                pdyn, dst, byimf, bzimf, g1, g2, ps,
                test_case['x'], test_case['y'], test_case['z']
            )
            
            # Run vectorized version
            bx_v, by_v, bz_v = extall_vectorized(
                options['iopgen'], 0, 0, 0, a, 43,
                pdyn, dst, byimf, bzimf, g1, g2, ps,
                np.array([test_case['x']]),
                np.array([test_case['y']]),
                np.array([test_case['z']])
            )
            
            # Compare results
            assert_close(bx_s, bx_v[0], test_case, options)
            assert_close(by_s, by_v[0], test_case, options)
            assert_close(bz_s, bz_v[0], test_case, options)
```

## Conclusion

The T01 vectorization requires careful handling of iterative algorithms, complex conditional logic, and multiple field components. By following this policy and building on the successful T96 vectorization patterns, we can achieve significant performance improvements while maintaining the high accuracy required for scientific applications.

Key success factors:
1. Robust handling of iterative convergence with multiple optimization strategies
2. Efficient region-based processing with proper masking
3. Careful numerical stability management
4. Comprehensive validation at each step
5. Clear documentation and testing
6. Support for both scalar and array-valued parameters
7. Full vectorization of harmonic calculations where possible

The vectorized T01 implementation will provide researchers with a high-performance tool for magnetospheric field calculations while preserving the model's scientific integrity. The policy provides multiple implementation options (masked vs full array operations, compiled vs pure NumPy) allowing developers to choose the best approach based on benchmarking results for their specific use cases.