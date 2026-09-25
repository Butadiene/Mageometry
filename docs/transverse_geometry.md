# Transverse magnetic geometry

[Documentation index](README.md) · [Geometry analysis](geometry_analysis.md)
· [Viewer guide](viewer.md) · [Input formats](simulation_data_formats.md)
· [Baseline FAC anisotropy theory](fac_anisotropy_theory.md)

The transverse API measures how the magnetic direction changes across a
field line. It returns local rotation and shear rates from first Cartesian
derivatives of B, together with the dimensionless rotation/shear balance eta.
The rates are per length; interpretation depends on
the field, grid resolution, and derivative step.

## Open a view

```bash
python examples/geometry_viewer.py --component beta_g --slice x --slice-only
python examples/geometry_viewer_simulation.py --xmf snapshot.xmf --component gamma --stride 4
```

The first command uses a T96 + dipole model. The second requires your own
file; replace `snapshot.xmf` with its path and choose an appropriate stride.
The simulation entry point has no built-in snapshot name or grid size.
Install `.[viz3d]` for viewing and `.[io,viz3d]` for XDMF/HDF5 input.
See the [viewer guide](viewer.md) for input layouts, slicing, colour scales,
units, and memory controls.

![Beta_g slice of an analytic magnetic field](images/transverse-beta-g.png)

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
assert np.isnan(rates['beta_g'][0])  # straight line on the central axis
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
with the broadcast shape. It contains exactly the seven keys below.

## Definitions and units

Following the [baseline theory](fac_anisotropy_theory.md), use
T=B/|B|, curvature normal n, b=T×n, and define
`p=b·∂nT`, `q=n·∂bT`, `a=n·∂nT`, and `d=b·∂bT`.

| Key | Definition | Interpretation |
| --- | --- | --- |
| `alpha` (α) | p−q = μ₀j∥/\|B\| | Twice the azimuthal mean winding rate |
| `beta_g` (β_g) | p+q = 𝒟/B | Signed off-diagonal shear in the Frenet frame |
| `delta_g` (δ_g) | a−d | Difference of transverse normal strains |
| `gamma` (Γ) | √(β_g²+δ_g²) | Basis-independent anisotropy; Γ/2 is the maximum angular-rate deviation |
| `omega_c` | sign(alpha) √max(alpha²−gamma²,0)/2 | Signed local coiling rate |
| `eta` | (alpha²−gamma²)/(alpha²+gamma²) | Dimensionless rotation/shear balance in [−1, 1] |
| `curvature` | \|(T·∇)T\| | Field-line curvature from first Cartesian derivatives |

The numerical implementation, analysis API, viewers, and CLIs all use
`beta_g` and `delta_g`. The symbol q denotes the matrix entry defined above.
The `delta` argument is the numerical difference step, separate from the
physical diagnostic `delta_g`.

All outputs except eta have inverse coordinate-length units. The five transverse
rates and eta are selectable in the overview viewer, are unaffected by
`current_scale`, and have no current arrows. The eigenvalue-based coiling
rate is a local diagnostic, not an integrated winding number or proof of a
flux rope; see [Tassev & Savcheva (2019)](https://arxiv.org/abs/1901.00865).

Eta is positive when alpha² exceeds gamma², negative when gamma² exceeds
alpha², and zero when the two nonzero magnitudes balance. It is **NaN**
when both are zero. Pure rotation gives +1 and pure shear/strain gives −1.
The viewer uses a fixed [−1, 1] colour scale by default. Select it with the
dropdown, F5/F6, or `--component eta`. Eta is a derived diagnostic for any
magnetic field source, including simulations; it is not a model input.

`omega_c` is zero when `alpha**2 <= gamma**2`; a finite zero is a defined
result, unlike NaN. When a Frenet normal exists, gamma equals
`sqrt(beta_g**2 + delta_g**2)`. Internally it is evaluated independently in
Cartesian coordinates, so it can remain finite when beta_g and delta_g are NaN.

## Validity and differences from the current API

The implementation uses seven B evaluations for Cartesian first derivatives,
then projects the gradient into the transverse plane. It never differentiates
n. Alpha, gamma, omega_c, and eta do not require a Frenet normal: straight lines
can retain nonzero anisotropy. Beta_g and delta_g are NaN when curvature is at or
below `curvature_tol`, including zero curvature. All results are NaN at
magnetic nulls or invalid stencils.
At weak curvature prefer gamma and examine derivative-step convergence.

Viewer alpha now uses this first-gradient estimate, so it remains available
on straight lines. The legacy `field_line_current_density()['alpha']` retains
its frame-difference method and stricter validity mask. The two estimates
agree in the converged limit where the frame is well defined, but need not
match to round-off. Likewise, legacy `B_twist_diff/|B|` estimates beta_g using
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


## Background gradient contributions

A current-free dipole can have nonzero gamma and eta = −1. To separate a
specified background gradient from total-field geometry, use the shared-frame
API; subtracting scalar gamma or eta does not perform this decomposition.

```python
import numpy as np
from mageometry import geopack, geopack_field
from mageometry.geometry import field_line_transverse_decomposition

ps = geopack.recalc(100.)
parmod = [2., -20., 0., -5., 0, 0, 0, 0, 0, 0]
total = geopack_field('t96', 'dip', parmod, ps)
background = geopack_field(None, 'dip', ps=ps)
parts = field_line_transverse_decomposition(total, background, -6., 2., 2., delta=.002)
np.testing.assert_allclose(parts['total']['shear'],
                           parts['background']['shear'] + parts['residual']['shear'],
                           atol=1e-13)
print(parts['total']['eta'], parts['residual']['eta'])
```

`reference` holds the total magnetic field, magnitude, tangent, normal,
binormal, projection and curvature. Each of `total`, `background`, `residual`
holds `gradient`, `transverse`, `shear`, `trace`, `divergence` and the six
transverse diagnostics. `gradient` uses component/derivative-direction order:
G_ij = ∂B_i/∂x_j. Vector/tensor dimensions follow the broadcast point dimensions;
scalar coordinates give vectors `(3,)`, tensors `(3, 3)` and scalar diagnostics.
Gradient and divergence have field/length units; transverse, shear, trace and
the five rates have inverse-length units, while eta is dimensionless.

Residual gradients are G_total − G_background. Both are projected with the
total P and divided by |B_total|. Tensors and signed alpha/beta_g/delta_g are
additive; norms, eta and omega_c are not. A background null is allowed, but
invalid background samples/stencils invalidate background and residual.
All projected results require valid total geometry. A zero residual gradient
has zero gamma and undefined eta. See the
[full derivation and limitations](fac_anisotropy_theory.md#background-attribution-in-the-total-field-frame).

This attribution retains background dependence through the total frame and
magnitude. In particular, a uniform perturbation has zero residual gradient
but can change total geometry. Residual eta/omega_c characterize the projected
gradient contribution, not the actual winding of residual-field lines. Call
`field_line_transverse_geometry` on an explicitly constructed residual field
if its own geometry is the intended question.

Choose **BACKGROUND** in `geometry_view` to select a preset or
**Load background file...**. The model CLI exposes **Dipole** on normal
startup (`python examples/geometry_viewer.py`); then choose
**CONTRIBUTION → Residual gradient** and **COMPONENT → eta/gamma**.
**None (total field)** disables attribution and restores all diagnostics.
Enabling a background from a current diagnostic selects eta automatically.
Background changes preserve total magnetic lines, cameras, slices and
per-diagnostic thresholds; shared limits are recalculated for the new
background. Invalid files or incompatible axes/declared units leave the
current view intact and show a message. Escape or Cancel exits file browsing.

Python callers can supply `background_choices={'Reference': background}` to
`geometry_view` or `transverse_contribution_view`. Presets must be explicit
callables or grids; metadata never implicitly selects a dipole. File loading
supports `.xmf`, `.xdmf`, `.vti`, `.vtr` and uses the CLI's total-input stride.
The Python API defaults to full file resolution; use `background_loader` for
a custom stride, region or field-array name and `background_directory` for
the initial folder. XDMF/HDF5 loading needs the optional `io` dependencies.
The file browser uses the existing VTK interface; no Tk/Qt installation is
needed. Bare HDF5 requires explicit grid coordinates: load it through Python
and register the result as a background preset.

Optional CLI arguments can preselect the corresponding comparison:

```bash
python examples/geometry_viewer.py --background dipole --component eta --contribution residual --slice x --slice-origin -6 0 0
python examples/geometry_viewer_simulation.py --xmf total.xmf --background-xmf background.xmf --component gamma
```

The second command needs your own matching files; `--background-vtk` accepts
a VTK background. Dipole background selection is restricted to the built-in
model demo. The Python entry point
`viz3d.transverse_contribution_view(total_grid, background, ...)` accepts a
callable or `GriddedField` as background, including data loaded from HDF5.
Declare it with `background_label`; optionally pass `field=total_callable`
for direct total evaluation, and `geometry_delta` for the shared difference
step. Grid-only inputs use equally coarsened preview interpolants; check
stride, preview resolution and step convergence for both inputs.

F7/F8 switches **Total field**, **Background gradient**, **Residual gradient**;
F5/F6 switches the six diagnostics. Magnetic lines remain those of the total
field, and switching contributions preserves cameras, slices, thresholds
and shared colour limits. Default thresholds and seeds use the total branch.
Eta defaults to [−1, 1]; other limits use the largest per-branch 98th percentile
of absolute values. `color_limits={'gamma': value}` overrides a shared limit.
Current arrows are absent. This comparison does not combine the dataset
selector of `compare_geometry` with the contribution selector.
