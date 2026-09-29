# Documentation image sources

[Documentation index](../README.md) · [Desktop workspace](../gui.md)
· [Standalone viewers](../viewer.md)

The figures in this directory include application renders of model or
analytic fields and original mathematical illustrations. The Qt workspace,
standalone PyVista viewers, and HTML design mockup
have different controls. Use a capture of the relevant application when
illustrating a workflow. The [mockup](../gui_mockup.html) contains schematic
diagrams and performs no magnetic calculations.

Run regeneration commands from the repository root. Inspect the selected
case, diagnostic, contribution, plane, units, colour limits, and visible
labels before replacing a referenced image. Grid and finite-difference
settings matter: matching colours across different capture sets do not
establish matching numerical results.

## Transverse-decomposition concept figure

The [flux-tube figure guide](../transverse_decomposition_figure.md) provides
English and Japanese SVG, PDF, and PNG downloads named
`transverse-decomposition.*` and `transverse-decomposition-ja.*`. These are
original mathematical illustrations, separate from application captures.
They show isolated θ, δ_g, β_g, and α modes with field lines generated
from exp(s M). The guide states the local thin-tube and frame assumptions,
parameters, and sources.

With `.[viz]`, run `python benchmark/transverse_decomposition_figure.py
--language both`. Japanese output also requires a CJK font. The generator
checks areas, strain axes, rotation, and the recovered transverse gradients.

## Qt desktop workspace

Requires `.[gui]` and a working desktop/OpenGL display. Run:

```bash
python benchmark/render_gui.py --output /tmp/mageometry-gui
```

| Asset | Generated file | View |
| --- | --- | --- |
| [gui-workspace.png](gui-workspace.png) | `workspace-all.png` | Full Qt window, All panels |
| [gui-slice.png](gui-slice.png) | `workspace-slice.png` | Same result, Slice focus |

The captures show residual-gradient eta for the By = −5 nT case in a
two-case T96 + dipole comparison, a 17 × 13 × 13 grid, direct 0.002 Re
derivatives, and the YZ plane at x = −6 Re. The background is a dipole at
the same epoch; attribution uses the total-field frame. See the
[complete conditions and copy commands](../gui.md#screenshots-and-regeneration).

Use each capture's matching `.session.json` to restore its recipe. The
script's final `session.json` belongs to a later restore check with different
settings. The `plot-*.png` exports omit Qt controls and are not substitutes
for these full-window images.

## Standalone T96 + dipole viewer

Requires `.[viz3d]`. Run `python benchmark/readme_screenshots.py`; use
`--output-dir /tmp/mageometry-screenshots` to inspect captures separately.
The [viewer guide](../viewer.md#regenerate-the-readme-screenshots) specifies
the model conditions and capture sequence: By = 0 nT, 65 × 49 × 49 input
nodes, 59 × 44 × 44 preview nodes, direct 0.002 Re derivatives, and a YZ
slice at x = −6 Re. These images show the standalone controls.

| Asset | Selected diagnostic / view |
| --- | --- |
| [fac-overview.png](fac-overview.png) | FAC overview, draggable 3D slice hidden; face-on panel visible |
| [fac-slice.png](fac-slice.png) | FAC overview, draggable 3D slice visible |
| [fac-slice-only.png](fac-slice-only.png) | Enlarged FAC slice |
| [current-components.png](current-components.png) | Normal current `mu0J_n`, overview |
| [current-components-slice.png](current-components-slice.png) | Binormal current `mu0J_b`, enlarged slice |
| [parallel-current-terms.png](parallel-current-terms.png) | First parallel contribution `B_dT_dn_b`, enlarged slice |
| [parallel-current-difference.png](parallel-current-difference.png) | Signed shear `B_twist_diff`, enlarged slice |
| [current-component-menu.png](current-component-menu.png) | Open diagnostic menu on the signed-shear slice |
| [viewer-eta-overview.png](viewer-eta-overview.png) | Total-field eta, overview |
| [viewer-eta-focus.png](viewer-eta-focus.png) | Total-field eta, enlarged slice |

## Standalone gradient contributions

With `.[viz3d]`, run `python benchmark/transverse_contribution_screenshots.py`.
The same By = 0 nT model and preview settings are used, with a dipole
background. All contribution captures retain the total-field lines, camera,
and YZ plane at x = −6 Re. Each diagnostic has a shared scale across branches.

| Asset | View |
| --- | --- |
| [viewer-background-menu.png](viewer-background-menu.png) | Open BACKGROUND menu before assigning a background |
| [contribution-eta-total.png](contribution-eta-total.png) | Total eta |
| [contribution-eta-background.png](contribution-eta-background.png) | Background-gradient eta in the total-field frame |
| [contribution-eta-residual.png](contribution-eta-residual.png) | Residual-gradient eta in the total-field frame |
| [contribution-gamma-total.png](contribution-gamma-total.png) | Total gamma |
| [contribution-gamma-background.png](contribution-gamma-background.png) | Background-gradient gamma in the total-field frame |
| [contribution-gamma-residual.png](contribution-gamma-residual.png) | Residual-gradient gamma in the total-field frame |

These are gradient attributions, not scalar differences or diagnostics of
independent residual magnetic fields. See the
[definitions](../transverse_geometry.md#background-gradient-contributions).

## Standalone dataset comparison

| Asset | View |
| --- | --- |
| [comparison-by-negative.png](comparison-by-negative.png) | Total-field alpha, By = −5 nT |
| [comparison-by-positive.png](comparison-by-positive.png) | Total-field alpha, By = +5 nT |

Both use the default six-case comparison, direct 0.002 Re derivatives,
a 59 × 44 × 44 preview, the YZ plane at x = −6 Re, and a scale shared
across all six cases. Regenerate with the
[comparison commands](../data_comparison.md#example-views), using `.[viz3d]`.

## Analytic transverse geometry

[transverse-beta-g.png](transverse-beta-g.png) shows B(x, y, z) = (−yz, xz, 1)
on the YZ plane at x = 1 grid unit. It uses 41³ nodes on [−2, 2]³ and direct
0.002 grid-unit derivatives. With `.[viz3d]`, run:

```bash
python benchmark/transverse_geometry_screenshot.py
```

Use `--output /tmp/mageometry-beta-g.png` to inspect a separate capture.
The [analytic example](../transverse_geometry.md#analytic-field-screenshot)
explains its units and distinguishes it from the T96 examples.

## README accuracy figures

The two Matplotlib figures under `benchmark/` are numerical validation
plots, independent of both viewer interfaces:

- [readme_validation_histogram.png](../../benchmark/readme_validation_histogram.png)
- [readme_validation_colormap.png](../../benchmark/readme_validation_colormap.png)

Regenerate with `python benchmark/readme_validation.py` after installing
`.[examples]`. If replacing them, update the associated metrics and runtime
environment in the [README](../../README.md#accuracy-validation) from the
same run. Their present values describe the recorded 2026-09-24 run.
