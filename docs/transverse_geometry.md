# Transverse magnetic geometry

[Documentation index](README.md) · [Geometry analysis](geometry_analysis.md)
· [Viewer guide](viewer.md) · [Input formats](simulation_data_formats.md)

The transverse API measures how the magnetic direction changes across a
field line. It returns local rotation and shear rates from first Cartesian
derivatives of B. These are rates per length; interpretation depends on
the field, grid resolution, and derivative step.

## Open a view

```bash
python examples/geometry_viewer.py --component sigma --slice x --slice-only
python examples/geometry_viewer_simulation.py --xmf snapshot.xmf --component gamma --stride 4
```

The first command uses a T96 + dipole model. The second requires your own
file; replace `snapshot.xmf` with its path and choose an appropriate stride.
The simulation entry point has no built-in snapshot name or grid size.
Install `.[viz3d]` for viewing and `.[io,viz3d]` for XDMF/HDF5 input.
See the [viewer guide](viewer.md) for input layouts, slicing, colour scales,
units, and memory controls.

![Sigma slice of an analytic magnetic field](images/transverse-sigma.png)

## Calculate rates without plotting

This complete example uses a synthetic field and needs only NumPy/SciPy:

```python
import numpy as np
from mageometry.geometry import field_line_transverse_geometry

def field(x, y, z):
    x, y, z = np.broadcast_arrays(x, y, z)
    return -y, x, np.ones_like(z)

x = np.array([0.0, 0.5, 1.0, 2.0])
y = np.zeros_like(x)
z = np.zeros_like(x)
rates = field_line_transverse_geometry(field, x, y, z, delta=0.002,
                                      curvature_tol=1e-8)
np.testing.assert_allclose(rates['alpha'], 2 / (x**2 + 1), rtol=1e-5)
assert np.isnan(rates['sigma'][0])  # straight line on the central axis
assert np.isfinite(rates['gamma'][0])
```

The function is also exported as `mageometry.field_line_transverse_geometry`.

| Parameter | Meaning |
| --- | --- |
| `field` | Callable accepting broadcast coordinates and returning `(bx, by, bz)` |
| `x`, `y`, `z` | Scalars or broadcast-compatible arrays in coordinate-length units |
| `delta` | Positive finite Cartesian step, scalar or three values; default 0.01 |
| `curvature_tol` | Nonnegative curvature cutoff in inverse length; default 0 |

The result is a dictionary of scalars for scalar coordinates, or arrays
with the broadcast shape. It includes all six keys below.

## Definitions and units

With T=B/|B|, curvature normal n, and b=T×n, define
`a=b·∂nT`, `c=n·∂bT`, `u=n·∂nT`, and `v=b·∂bT`.

| Key | Definition | Interpretation |
| --- | --- | --- |
| `alpha` | a−c = μ₀j∥/|B| | Twice the azimuthal mean winding rate |
| `sigma` | a+c | Signed off-diagonal shear in the Frenet frame |
| `q` | u−v | Difference of transverse normal strains |
| `gamma` | √(sigma²+q²) | Basis-independent anisotropy; gamma/2 is the maximum angular-rate deviation |
| `omega_c` | sign(alpha) √max(alpha²−gamma²,0)/2 | Signed local coiling rate |
| `curvature` | \|(T·∇)T\| | Field-line curvature from first Cartesian derivatives |

All six outputs have inverse coordinate-length units. The five transverse
rates are selectable in the overview viewer, are unaffected by
`current_scale`, and have no current arrows. The eigenvalue-based coiling
rate is a local diagnostic, not an integrated winding number or proof of a
flux rope; see [Tassev & Savcheva (2019)](https://arxiv.org/abs/1901.00865).

`omega_c` is zero when `alpha**2 <= gamma**2`; a finite zero is a defined
result, unlike NaN. When a Frenet normal exists, gamma equals
`sqrt(sigma**2 + q**2)`. Internally it is evaluated independently in
Cartesian coordinates, so it can remain finite when sigma and q are NaN.

## Validity and differences from the current API

The implementation uses seven B evaluations for Cartesian first derivatives,
then projects the gradient into the transverse plane. It never differentiates
n. Alpha, gamma, and omega_c do not require a Frenet normal: straight lines
can retain nonzero anisotropy. Sigma and q are NaN when curvature is at or
below `curvature_tol`, including zero curvature. All results are NaN at
magnetic nulls or invalid stencils.
At weak curvature prefer gamma and examine derivative-step convergence.

Viewer alpha now uses this first-gradient estimate, so it remains available
on straight lines. The legacy `field_line_current_density()['alpha']` retains
its frame-difference method and stricter validity mask. The two estimates
agree in the converged limit where the frame is well defined, but need not
match to round-off. Likewise, legacy `B_twist_diff/|B|` estimates sigma using
the old discretization. `B_twist_diff` no longer draws current arrows because
it represents shear, even though it has current-like units.

`viz.plot_geometry_map(..., 'alpha')` and the general slice/line plotters
still use the legacy current estimate. Supply a custom quantity callable
when you want the transverse estimate in those plotters; the overview's
`viz3d.geometry_view`, `current_view`, and component-enabled `fac_view`
use the first-gradient estimate.

## Derivative resolution

For simulation grids the viewer differentiates the masked preview's linear
interpolant at the smallest preview spacing by default. Use
`--geometry-delta` and change `--stride` / `--max-points` to check convergence;
preview coarsening changes derivative estimates. Magnetic context lines,
camera and slice positions remain fixed when switching quantities.

For direct API calls, choose `delta` in the input coordinate units. For a
smooth model, compare smaller steps. For gridded data, compare interpolation
and grid resolution too; increasing `curvature_tol` masks unresolved normals
but does not repair their values. The viewer currently does not expose
`curvature_tol` as a CLI option. See
[step-size and validity guidance](geometry_analysis.md#undefined-geometry-and-step-size).
