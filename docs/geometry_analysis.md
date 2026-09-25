# Geometry analysis

[Documentation index](README.md) · [Input formats](simulation_data_formats.md)
· [Viewer guide](viewer.md) · [Transverse geometry](transverse_geometry.md)

`mageometry.geometry` is the primary analysis API. Its public functions are
also exported from `mageometry`. They accept a field callable and Cartesian
coordinates; no geopack initialization is needed for a custom field or
`GriddedField.field()`.

## Field and coordinate contract

A field callable accepts broadcast NumPy coordinates and returns three
magnetic components in the same Cartesian frame. Use a consistent,
right-handed coordinate system. Field and length units are caller-defined.
Scalar coordinates produce scalar geometry values; array coordinates
produce values with the broadcast shape.

The examples in this page run in order and use a synthetic helical field:

```python
import numpy as np
from mageometry import (
    field_line_curvature, field_line_torsion, field_line_frenet_frame,
    field_line_frame_quality, field_line_directional_derivatives,
    field_magnitude_derivatives, field_line_current_density,
    field_aligned_current_density, field_line_transverse_geometry,
    trace_field_lines,
)

def field(x, y, z):
    x, y, z = np.broadcast_arrays(x, y, z)
    return -y, x, np.ones_like(z)

x = np.array([0.5, 1.0, 2.0])
y = np.zeros_like(x)
z = np.zeros_like(x)
delta = 1e-3
```

For geopack fields, call `geopack.recalc(ut)` first and construct
`geopack_field(external='t96', internal='dip', parmod=..., ps=...)` with
your model parameters. Positions then use Re and fields use nT. `recalc`
updates shared geopack state, so a callable with an internal field uses
the current internal-field epoch when evaluated.

## Frames, curvature, and torsion

```python
kappa = field_line_curvature(field, x, y, z, delta=delta)
tau = field_line_torsion(field, x, y, z, delta=delta)
tx, ty, tz, nx, ny, nz, bnx, bny, bnz, curvature = field_line_frenet_frame(
    field, x, y, z, delta=delta)

np.testing.assert_allclose(kappa, x / (x**2 + 1), rtol=1e-5)
np.testing.assert_allclose(tau, 1 / (x**2 + 1), rtol=1e-5)
```

T is the unit tangent B/|B|, n the principal curvature normal, and b=T×n
the binormal. Lowercase b here denotes a frame vector, not the magnetic
field. The normal is formed from the perpendicular part of the estimated
dT/ds. Curvature and torsion have inverse-length units.

| Function | Result |
| --- | --- |
| `field_line_tangent` | `(tx, ty, tz)` |
| `field_line_normal` | `(nx, ny, nz)` |
| `field_line_binormal` | `(bnx, bny, bnz)` |
| `field_line_curvature` / `field_line_torsion` | Scalar or array |
| `field_line_frenet_frame` | Nine vector components followed by curvature |
| `field_line_geometry_complete` | Same ten entries followed by torsion |
| `field_line_frame_quality` | `abs(T·dT/ds) / abs(dT/ds)` |

The last result is a finite-difference consistency diagnostic. It is NaN
when dT/ds is zero or undefined. It is not a bound on the total numerical
error. Frame functions reject normals with a diagnostic above
`orthogonality_tol` (default 0.1); tangent and curvature have separate
validity conditions.

## Directional derivatives

```python
derivatives = field_line_directional_derivatives(field, x, y, z, delta=delta)
curvature_from_frame = derivatives['dT_dT_n']
torsion_from_frame = derivatives['dn_dT_b']
```

Keys describe `(partial_u v)·w`: for example, `dT_dn_b` is the derivative
of T along n projected onto b. The dictionary contains **18 projections**:
nine independent components plus nine antisymmetric partners for
validation. See the [README's nine-component example](../README.md#field-line-directional-derivatives)
for their grouping.

These calculations need frames at neighbouring stencil points, not only
at the requested coordinate. All derivative projections at a point are
masked when any required frame is invalid or when the normal changes too
much across a stencil. The default test requires
`n(plus)·n(minus) > normal_flip_tol`, with `normal_flip_tol=0.9`.
`verify_antisymmetry_relations(derivatives)` reports finite-difference
residuals; it does not replace a step-size check.

## Magnetic magnitude and current density

```python
mag = field_magnitude_derivatives(field, x, y, z, delta=delta)
current = field_line_current_density(field, x, y, z, delta=delta)
fac = field_aligned_current_density(field, x, y, z, delta=delta)
rates = field_line_transverse_geometry(field, x, y, z, delta=delta)

# This synthetic field has curl(B) = (0, 0, 2).
np.testing.assert_allclose(fac, 2 / np.sqrt(x**2 + 1), rtol=1e-5)
np.testing.assert_allclose(rates['alpha'], 2 / (x**2 + 1), rtol=1e-5)
```

| API | Main outputs | Units |
| --- | --- | --- |
| `field_magnitude_derivatives` | `B`, `dB_dT`, `dB_dn`, `dB_db` | Field; derivatives in field/length |
| `field_line_current_density` | `mu0J_T/n/b`, `mu0J_x/y/z`, parallel terms, `B_twist_diff`, `alpha`, `B`, `curvature` | Currents in field/length; alpha and curvature in 1/length |
| `field_aligned_current_density` | Cartesian curl(B)·T | Field/length |
| `field_line_transverse_geometry` | `alpha`, `beta_g`, `delta_g`, `gamma`, `omega_c`, `eta`, `curvature` | 1/length except dimensionless eta |

The current API returns **μ₀J**, not J in SI units. In particular,
`B_dT_dn_b + B_dn_db_T = mu0J_T`, while `B_twist_diff` is their signed
difference. Both parallel terms describe contributions along T. For nT
and Re input, multiply current values by approximately 0.125 to express
J in nA/m²; other input units require their own conversion.

The Frenet reconstruction needs a valid normal, including for its Cartesian
outputs and `alpha`. The independent Cartesian FAC calculation does not.
Transverse `alpha` uses first derivatives of B and is also defined on
straight lines. General named-quantity plotters (`viz.plot_geometry_map`,
`viz3d.slice_view`) use the legacy current API's `alpha`; the geometry
overview uses transverse `alpha`. See [transverse definitions](transverse_geometry.md)
for the validity and interpretation of each rate.

`verify_divergence_identity(field, x, y, z, delta=...)` returns the
Frenet-frame estimate of ∇·B. `grid.divergence(relative=False)` evaluates
Cartesian differences on grid nodes. Neither is expected to be exactly
zero for an arbitrary empirical field or interpolant.

## Undefined geometry and step size

| Location or condition | Expected behavior |
| --- | --- |
| Magnetic null or undefined field | Tangent and dependent geometry are NaN |
| Straight, nonzero field | Curvature can be zero; n, b, torsion, and Frenet currents are undefined |
| Invalid neighbouring samples | Quantities needing those stencils become NaN |
| Zero or weak curvature | Inspect transverse `gamma`; `beta_g` and `delta_g` need a resolved normal |

Check each result with `np.isfinite`. Do not replace undefined values with
zero before averaging or plotting. Use a positive `delta` in coordinate
units (default 0.01 for these analysis functions). Most Frenet APIs use a
scalar step along frame directions; Cartesian FAC and transverse geometry
also accept three Cartesian step sizes.

```python
estimates = np.array([
    field_line_curvature(field, 1.0, 0.0, 0.0, delta=step)
    for step in (0.02, 0.01, 0.005)
])
quality = field_line_frame_quality(field, x, y, z, delta=delta)
print(estimates, quality)
```

For a smooth analytic field, reducing the step should initially reduce
truncation error; eventually cancellation and input precision matter.
For interpolated data, vary grid resolution and interpolation as well as
`delta`. A smaller step cannot recover information absent from the grid.
See [interpolation guidance](simulation_data_formats.md#interpolation-and-the-finite-difference-step).

## Trace and analyze the same field

```python
trace = trace_field_lines(field, [1.0, 2.0], [0.0, 0.0], [0.0, 0.0],
                         direction='both', ds=0.05, max_steps=100)
line_x, line_y, line_z = trace.path(0)
arc_length = trace.arc_length(0)
line_curvature = field_line_curvature(field, line_x, line_y, line_z, delta=delta)
assert arc_length[trace.start_index[0]] == 0
```

`direction=1` follows B; `-1` follows −B. With `'both'`, paths run from the
−B end to the +B end and arc length is signed around the seed.
`trace.status` describes the +B end; `trace.status_backward` describes
the −B end. Codes are 0 (inner sphere), 1 (outer sphere or box), 2 (step
limit), 3 (undefined field), and 4 (custom stop). This example has no
spatial boundary, so it terminates at the step limit.

For simulation fields, pass `bounds=grid.bounds` and retain the default
NaN out-of-domain fill. `ds` is a nominal integration step; it is separate
from the geometry derivative step `delta`. The engine tracer
`geopack.trace_vectorized` has different return values and the opposite
direction-sign convention; see its [reference table](../README.md#field-line-tracing-1).
