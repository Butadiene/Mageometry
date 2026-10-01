# Documentation image sources

[Documentation index](../README.md) · [Desktop workspace](../gui.md)
· [Standalone viewers](../viewer.md)

The figures in this directory include application renders of model or
analytic fields and original mathematical illustrations. The Qt workspace and
standalone PyVista viewers have different controls. Use a capture of the
relevant application when illustrating a workflow.

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

## Directional-rotation concept figure

The [directional-rotation guide](../directional_rotation_figure.md) provides
English and Japanese SVG, PDF and PNG files named `directional-rotation.*`
and `directional-rotation-ja.*`. This original mathematical illustration
links 3D neighbours, signed angular arrows and Ω(φ), including Ω_n = p,
Ω_b = −q, the FAC sum, signed-shear difference and mean α/2. It uses the
same zero-torsion reference curve as the decomposition figure and specifies
the general Frenet correction dφ/ds = Ω − τ.

With `.[viz]`, run `python benchmark/directional_rotation_figure.py --language
both`. The guide lists the illustrative coefficients, scales and numerical
checks. No external image or simulation data is used.

## Qt desktop workspace

`gui-alpha-gradient.png` shows `(dalpha/ds)/|B|` in the desktop plot layout
(without the Qt control panels), with a YZ slice at x = -6 Re. It uses a
25 × 19 × 19 T96 + dipole grid, By = 0 nT, Dst = -20 nT, Bz = -5 nT,
and direct differences of 0.002 Re. Reproduce it without Qt using:

```bash
python -m mageometry.gui --shape 25 19 19 --component dalpha_ds_over_B \
  --slice x --slice-origin -6 0 0 --layout three_d_slice \
  --screenshot /tmp/mageometry-alpha-gradient.png
```

Copy the resulting PNG to `docs/images/gui-alpha-gradient.png`. The adjacent
session JSON records the complete parameters and camera state for inspection.

Full-window captures require `.[gui]` and a working desktop/OpenGL display. Run:

```bash
python benchmark/render_gui.py --output /tmp/mageometry-gui
```

The same script exercises model selection with Qt keyboard events and writes
`source-t89.png`, `source-t01.png` and `source-t04.png`. Copy these to
`gui-source-t89.png`, `gui-source-t01.png` and `gui-source-t04.png` here.
They show the actual source dialog with each model's demonstration defaults;
they do not depict an inferred storm history.

| Asset | Generated file | View |
| --- | --- | --- |
| [gui-workspace.png](gui-workspace.png) | `workspace-all.png` | Full Qt window, All panels |
| [gui-three-d-slice.png](gui-three-d-slice.png) | `workspace-three_d_slice.png` | Same result, 3D + Slice |
| [gui-display.png](gui-display.png) | `workspace-display.png` | Same result, Display tab grouped into Shared, 3D and peak maps, and Slice sections |
| [gui-display-positive.png](gui-display-positive.png) | `workspace-display-positive.png` | Same result and cameras, positive values only |
| [gui-display-negative.png](gui-display-negative.png) | `workspace-display-negative.png` | Same result and cameras, negative values only |
| [gui-display-ranges.png](gui-display-ranges.png) | `workspace-display-ranges.png` | Interval [−0.5, 0.5], slice colours [−0.6, 0.9], y [−5, 5] Re, z [−4, 4] Re |
| [gui-no-trace.png](gui-no-trace.png) | `workspace-no-trace.png` | Second case (By = +5 nT), 0.001 Re differences, tracing disabled in Analysis |
| [gui-slice.png](gui-slice.png) | `workspace-slice.png` | Same result, Slice focus |

The captures show residual-gradient eta for the By = −5 nT case in a
two-case T96 + dipole comparison, a 17 × 13 × 13 grid, direct 0.002 Re
derivatives, and the YZ plane at x = −6 Re. The background is a dipole at
the same epoch; attribution uses the total-field frame. See the
[complete conditions and copy commands](../gui.md#screenshots-and-regeneration).
The captures were refreshed on 2026-09-30 with retained 3D illumination,
smooth region shading and translucent context lines. The capture script
also verifies mouse rotation through 3D + Slice and 3D/Slice focus layout changes,
numeric threshold entry, manual slider range entry, slider keyboard input,
positive/negative value selection, signed intervals, manual slice ranges,
and trace disablement followed by Apply and session restore.

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
The capture scripts use `mageometry.session.presets.model_snapshot`, the
same model preset as `python -m mageometry.viz3d`; the preset preserves the
original single-case units, metadata labels, mask and automatic trace seeds.

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
[comparison commands](../viewer.md#comparison-screenshots), using `.[viz3d]`.

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
