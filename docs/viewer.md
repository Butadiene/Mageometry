# Standalone viewer guide

For Japanese explanations of the shared diagnostics and the Qt desktop
controls, see [3D viewer 日本語詳細ガイド](gui_ja.md).

[Documentation index](README.md) · [Input formats](simulation_data_formats.md)
· [Geometry analysis](geometry_analysis.md) · [Transverse definitions](transverse_geometry.md)

The 3D overview displays magnetic geometry and current components on a
rectilinear grid. It combines spatial regions, magnetic context lines,
signed peak projections, and movable slices.

For model/file setup, combined case/contribution comparison, four display
layouts, and session save/restore in one window, use the optional
[desktop workspace](gui.md). Use the standalone CLI below for command-line
source selection, off-screen PNGs, or a PyVista window without Qt.

## Install and choose an entry point

Run commands from the repository root after installing PyVista. XDMF/HDF5
input also needs h5py:

```bash
python -m pip install -e '.[io,viz3d]'
python -m mageometry.viz3d --component gamma
```

The common entry point is `python -m mageometry.viz3d`; `mageometry-viewer`
is the installed command. Both accept the model, file, background, comparison and screenshot
options described here. Run `--help` for the full option list.

| Input | Initial view |
| --- | --- |
| No source options | T96 + dipole, By = 0 nT, `alpha` |
| `--xmf`, `--vtk`, or `--h5` | Supplied snapshot, `alpha` |
| `--by -5 -3 -1 1 3 5` | Six T96 cases, `alpha`, YZ slice at x = −6 Re |
| Background option | Total/background/residual gradients, `eta` |

The model demonstration uses Re and nT and converts displayed currents to
nA/m². File inputs retain their native units. `--shape NX NY NZ` sets the
model display grid; `--evaluation grid` selects interpolated derivatives
and traces for a model. File sources always use grid evaluation.

For desktop launch presets, see the [example index](../examples/README.md#desktop-example-launchers).
For multiple datasets, see [comparison in Python](#compare-datasets-in-python)
and the shared settings below.

## Supply your file

```bash
python -m mageometry.viz3d --xmf snapshot.xmf --stride 4
python -m mageometry.viz3d --xmf snapshot.xmf --h5 field.h5
python -m mageometry.viz3d --vtk snapshot.vti
python -m mageometry.viz3d --h5 field.h5 --origin 0 0 0 --spacing 1 1 1
```

Replace filenames, origin, and spacing with your own values. Relative CLI
paths are resolved from the working directory. HDF5 paths referenced inside
an XDMF file are resolved relative to that XDMF file. `--h5` with `--xmf`
overrides the heavy-data filename while keeping the metadata's dataset
paths. `--vtk` cannot be combined with `--xmf` or `--h5`.

| Input | Required layout in the CLI |
| --- | --- |
| XDMF | Uniform grid; scalar attributes `BX`, `BY`, `BZ`; HDF5 heavy data |
| HDF5 | Datasets `BX`, `BY`, `BZ` stored `(nz, ny, nx)`; explicit `--origin` and `--spacing` in x/y/z order |
| VTK | ImageData or RectilinearGrid (`.vti`/`.vtr`) with vector array `B` |

The CLI reads one snapshot. For a time series, load a step with
`load_xdmf_series` in Python and pass that grid to the viewer. For other
array names or HDF5 axis order, use the Python reader options:

```python
from mageometry import load_hdf5, viz3d

# Example layout: replace names, origin, and spacing with your file's values.
grid = load_hdf5('field.h5', datasets=('Bx', 'By', 'Bz'),
                 origin=(0, 0, 0), spacing=(1, 1, 1), zyx_order=False)
viz3d.geometry_view(grid, component='gamma', slice_panel=True)
```

This assumes `(nx, ny, nz)` arrays. `load_xdmf(..., components=(...))` and
`load_vtk(..., name=...)` provide the corresponding name overrides.
See [exact format requirements](simulation_data_formats.md#part-iii--bundled-readers-xdmf--hdf5).

## Select a diagnostic and a slice

```bash
python -m mageometry.viz3d --xmf snapshot.xmf --component beta_g --slice x --slice-only
python -m mageometry.viz3d --component mu0J_b --slice x --slice-origin -6 0 0 --slice-only
```

`--slice x` means a plane **normal to x**, so it displays YZ. `--slice-origin`
specifies a point on the plane in grid coordinates, not a camera position.
With no slice direction specified, a single snapshot starts on XZ.
By comparisons start on YZ at x = −6 Re.

| Control | Action |
| --- | --- |
| Dropdown / F5 / F6 | Select / previous / next diagnostic |
| F1 / F2 / F3 | Align the slice with YZ / XZ / XY |
| F4 | Toggle the enlarged face-on slice; its slider moves the plane |
| `c` | Toggle the draggable slice in the main 3D view |
| `x` / `y` / `z` | View along a coordinate axis |
| `r` | Reset the overview camera, or fit the slice in enlarged mode |
| `l` / `a` / `s` | Toggle magnetic lines / current arrows / spatial regions |
| Shift+drag / wheel | Pan / zoom |

Component changes keep the camera, plane position, and magnetic context
lines fixed. Each component remembers its own threshold. Dropdown keyboard
selection uses Up/Down and Enter; Esc closes the menu.
The shear diagnostics are `beta_g` (β_g) and `delta_g` (δ_g), using the
same names as the numerical implementation and analysis API. Their
definitions and interpretation are in the [baseline theory](fac_anisotropy_theory.md).

### Source information

The header shows declared source information in both the overview and the
enlarged F4 view, and follows the selected comparison dataset. The bundled
T96 examples supply the model name, Pdyn, Dst, IMF By/Bz, dipole tilt (rad),
and epoch (Unix seconds). Text wraps and fits the available header space;
the fitted view leaves room for it above the plot.

This is a generic provenance display, separate from model evaluation. Supply
`GriddedField.metadata` with optional `model`, `source`, `time`,
`coordinate_system`, `length_unit`, `field_unit`, and a `parameters` mapping
whose labels include units. For example, a simulation can declare:

```python
grid.metadata.update(source='run/step.vti', time=120.,
                     parameters={'Resistivity [native]': 0.01, 'Step': 240})
```

No model inputs are inferred from a callable, and this information does not
alter calculations or convert units. Callers must keep metadata consistent
with their supplied grid/field. Missing metadata leaves the header empty.

![Standalone T96 plus dipole viewer: total-field eta with source information and a YZ slice at x = -6 Re](images/viewer-eta-overview.png)

[Enlarged slice with the same source information](images/viewer-eta-focus.png).

![Standalone analytic-field viewer: beta_g on a YZ slice at x = 1 grid unit](images/transverse-beta-g.png)

This separate analytic example uses B(x, y, z) = (−yz, xz, 1), not the T96
model above. Its [capture recipe](transverse_geometry.md#analytic-field-screenshot)
specifies the grid, derivative step, and regeneration command.

## Read the colours, arrows, and projections

Both desktop and standalone launchers support `--model t89`, `t96` (default),
`t01`, and `t04`, each combined with the internal dipole. T89 uses `--iopt`,
T01 adds `--g1`/`--g2`, and T04 adds `--w W1 W2 W3 W4 W5 W6`.
See the [model inputs and examples](gui.md#model-and-file-workflows).

| Component group | Meaning | Arrows |
| --- | --- | --- |
| `fac` | Independent Cartesian curl(B)·T | Along/against T |
| `mu0J_T`, `B_dT_dn_b`, `B_dn_db_T` | Parallel current and its two frame contributions | T |
| `mu0J_n`, `mu0J_b` | Normal and binormal current | n or b |
| `mu0J_x`, `mu0J_y`, `mu0J_z` | Cartesian components of the Frenet reconstruction | x, y, or z |
| `B_kappa`, `minus_dB_dn` | Curvature and magnitude-gradient terms of `mu0J_b` | b |
| `B_twist_diff` | D = Bβ_g, the signed difference of the two parallel terms | None |
| `alpha`, `beta_g`, `delta_g`, `gamma`, `omega_c` | Transverse rates; see [definitions](transverse_geometry.md) | None |
| `dalpha_ds` | T·∇α: alpha gradient along B, in inverse length squared | None |
| `dalpha_ds_over_B` | (T·∇α)/B, in inverse field unit / length squared | None |
| `dfac_ds` | T·∇FAC: signed parallel-current gradient, in FAC units / length | None |
| `eta` | (alpha²−gamma²)/(alpha²+gamma²), dimensionless rotation/shear balance | None |

Red and blue indicate the sign in the selected basis. Only T-directed
components represent current along or against B. `B_twist_diff` is a
shear diagnostic with field/length units; it is not total parallel current.
When current conversion is enabled, its displayed label is `D / mu0`.

The along-field derivatives use arc length in the direction **T = B/|B|**.
A positive derivative means the signed scalar increases along B; it does
not specify the current direction. Only `dfac_ds` receives `current_scale`.
The two alpha derivatives remain in native geometry/field units. Magnetic
units for their labels come from the grid's `field_unit` metadata, with
`field unit` as the fallback. Gradient attribution supports the original
six transverse diagnostics only.

For direct fields, nested central differences evaluate alpha and FAC at
`r ± geometry_delta*T(r)`. The inner alpha curl uses `geometry_delta`; the
inner FAC curl uses `delta` (the desktop's optional FAC override). For grid
evaluation, the viewer takes central Cartesian differences of the displayed
preview alpha/FAC arrays and projects onto T, using the actual, possibly
nonuniform axis spacing. The grid alpha calculation still uses
`geometry_delta`. Required missing samples, magnetic nulls, and incomplete
boundary stencils remain NaN. These quantities involve second magnetic-field
derivatives, so check both difference-step and grid-resolution convergence.
For exact derivatives, `∂s FAC = B ∂s alpha + alpha ∂s B` in native units;
`dfac_ds` is not generally `B*dalpha_ds`.

```bash
python -m mageometry.viz3d --xmf snapshot.xmf \
  --stride 4 --component dalpha_ds_over_B --slice x --slice-only
```

Use `--component dalpha_ds` or `--component dfac_ds` for the other derivatives.
The numerical callable API is
`mageometry.geometry.field_aligned_current_derivatives(field, x, y, z, delta=..., fac_delta=...)`.

![Along-field alpha derivative divided by magnetic strength, with a YZ slice at x = -6 Re](images/gui-alpha-gradient.png)

The [capture recipe](images/README.md#qt-desktop-workspace) specifies the T96
parameters and numerical settings for this desktop plot export.

The overview maps select the signed value with the largest absolute
magnitude along each sightline. They are peak projections, not slices or
integrated currents. Thresholding affects regions, arrows, and peak-map
visibility; slices show all finite values. Undefined nodes and cells remain
blank. Slice dragging reuses cached values.

Colour limits use the 98th percentile of each component's absolute value
and stay fixed while adjusting its threshold. Compare legends across
components: identical colours need not represent identical amplitudes.
Nonnegative `gamma` uses the positive half of the diverging scale.
Eta uses a fixed [−1, 1] scale by default; alpha = gamma = 0 is undefined
and remains blank. Like the transverse rates, eta is never current-scaled.
The comparison viewer uses the largest per-case percentile to fix a common
range before displaying that diagnostic; it does not rescale on selection.

## Resolution, memory, and derivatives

| Option | Effect |
| --- | --- |
| `--stride N` | Keep every Nth input node; default 1 |
| `--max-points N` | Limit the displayed/analysed preview; default 120000, minimum 27 |
| `--geometry-delta H` | Derivative step for transverse rates and Frenet components, in coordinate units |
| `--threshold VALUE` | Initial absolute cutoff in the displayed component's units |
| `--no-trace` | Skip field-line tracing; diagnostic fields, slices and regions are still computed |

Retain at least three nodes on each axis. Reader stride changes which nodes
are loaded; preview coarsening happens afterwards and does not reduce the
initial file-loading cost. XDMF/HDF5 readers subset during loading, whereas
VTK is read in full before subsetting. Python also offers reader `region=`
and viewer `max_points=None` (the CLI expects an integer budget).

Grid-only `fac` uses Cartesian differences with the preview's axis spacing;
its boundary nodes are blank. Other components use the masked preview's
linear interpolant with `geometry_delta` equal to the smallest preview
spacing by default. With an explicit `field` and `delta`, it defaults to
`min(delta)`. `--geometry-delta` does not alter the independent `fac`
estimate. Compare multiple grid resolutions and steps when checking small
features; see [numerical guidance](geometry_analysis.md#undefined-geometry-and-step-size).
In comparisons, supplying per-case `fields` uses the direct-evaluation
path with an explicit shared `delta`. Display-grid coarsening then changes
the sampled diagnostic locations, without changing the magnetic field used
for derivatives or tracing. See [direct model evaluation](#direct-model-evaluation-in-python).

## Python API, native units, and screenshots

This standalone synthetic example needs no simulation file:

```python
import numpy as np
from mageometry import GriddedField, viz3d

axes = (np.linspace(-2, 2, 17),) * 3
x, y, z = np.meshgrid(*axes, indexing='ij')
grid = GriddedField(*axes, -y, x, np.ones_like(z))
plotter = viz3d.geometry_view(grid, component='gamma', slice_normal='x',
                             slice_panel=True, n_lines=4, show=False)
plotter.show()
```

`geometry_view` starts with `alpha`, `current_view` with `mu0J_T`, and
`fac_view` with `fac`. `geometry_view` and `current_view` enable the
component selector; plain `fac_view` keeps FAC-only controls. In the Python
API, `slice_panel=False` is the default; the CLI enables it. An existing
`plotter=` supports a single scene, so omit the companion panels there.

Viewer labels and `GriddedField.metadata` do not convert data. Native current
values represent μ₀J in field/length units. If B is in nT and coordinates in
Re, `current_scale=0.125`, `current_unit='nA/m^2'`, and `length_unit='Re'`
label the approximately converted currents. The five transverse rates are
never current-scaled and retain inverse-length units; eta is dimensionless.
`planet_radius`
draws a reference sphere; use `mask=` to exclude its interior from analysis.

For an off-screen PNG:

```bash
python -m mageometry.viz3d --xmf snapshot.xmf --component gamma --slice x --slice-only --screenshot geometry.png
```

For Python screenshots, set `pyvista.OFF_SCREEN = True` before constructing
the plotter, pass `show=False`, then call `plotter.screenshot(path)` and
`plotter.close()`.

## Compare datasets in Python

`compare_geometry(cases, ...)` accepts an ordered mapping of labels to
`GriddedField` snapshots. **DATASET** (F7/F8) and **COMPONENT** (F5/F6) are
independent, including in the F4 enlarged slice. For example:

```bash
python -m mageometry.viz3d --by -5 -3 -1 1 3 5
python -m mageometry.viz3d --by -5 -3 -1 1 3 5 --initial-by 5 --slice-only
python -m mageometry.viz3d --by -5 -3 -1 1 3 5 --evaluation grid
python -m mageometry.viz3d --by -5 -3 -1 1 3 5 --color-limit 0.2 --screenshot comparison.png
```

The CLI creates T96 + dipole grids at epoch 100 Unix seconds, Pdyn = 2 nPa,
Dst = −20 nT and IMF Bz = −5 nT, with GSM coordinates in Re and fields in nT.
`--initial-by` must occur in `--by`. `--color-limit` and `--threshold`
apply to the initial diagnostic. Default display sampling is 65 × 49 × 49
on x = [−15, 5], y/z = [−8, 8] Re, with a 120000-node preview budget.
The inner r < 2.5 Re region is masked. Currents use the 0.125 conversion.
The five-case [desktop preset](../examples/README.md#desktop-example-launchers)
uses different Dst/Bz/By conditions; reproduce comparisons with explicit inputs.

Direct evaluation uses a 0.002 Re Cartesian FAC step (`--delta`), also the
default geometry step; `--geometry-delta` overrides geometry alone.
Grid mode rejects `--delta`: FAC uses preview-axis differences and geometry
defaults to the smallest preview spacing. `--shape` and `--max-points`
together control the sampled display resolution. Model callables restore
their epochs on every evaluation; evaluation remains serial because geopack
has shared state. For combined dataset/background selection and saved recipes,
use the [desktop workspace](gui.md).

### Direct model evaluation in Python

The same T96 helpers are available from the installed package:

```python
from mageometry.session.presets import make_cases, make_fields, inner_mask
from mageometry import viz3d

by_values = (-5., 0., 5.)
viz3d.compare_geometry(
    make_cases(by_values), fields=make_fields(by_values), delta=0.002,
    mask=inner_mask, component='alpha', length_unit='Re',
    current_scale=0.125, current_unit='nA/m^2', slice_panel=True)
```

`fields` is a mapping with exactly the same labels as `cases`. Each value
accepts broadcast coordinates and returns `(bx, by, bz)` in the corresponding
grid's coordinates and units. All cases must supply a callable, and an
explicit positive shared `delta` is required (a scalar or three Cartesian
steps). The default geometry step is `min(delta)`; `geometry_delta` can
override it. A single shared `field` is rejected to avoid assigning the
wrong model to a case. Omitting `fields` keeps the grid-data workflow.

The callables supply field values at derivative stencil points and along
traces. Input grids still define the display coordinates and base validity
mask: a NaN input node stays blank even when its callable is defined there.
Callables must represent the same magnetic field as their grids and remain
reproducible after evaluating another case. The viewer does not interpret
model names, parameters, or epochs. Stateful models need an adapter such as
the shared model source evaluator.

### Supply model grids or file data

This small standalone example requires no files:

```python
import numpy as np
from mageometry import GriddedField, viz3d

axis = np.linspace(-2, 2, 25)
x, y, z = np.meshgrid(axis, axis, axis, indexing='ij')
cases = {}
for strength in (0.5, 1.0, 1.5):
    cases[f'Twist = {strength:g}'] = GriddedField(
        axis, axis, axis, -strength*y, strength*x, np.ones_like(x),
        metadata={'coordinate_system': 'Cartesian',
                  'length_unit': 'm', 'field_unit': 'T'})

viz3d.compare_geometry(cases, component='alpha', length_unit='m',
                       slice_panel=True, slice_normal='z',
                       color_limits={'alpha': 3.0}, cache_size=2)
```

For files, construct the mapping with the existing readers:

```python
from mageometry import load_xdmf, viz3d

# Replace these placeholders with your files; use identical reader settings.
cases = {'Run A': load_xdmf('run_a.xmf', stride=4),
         'Run B': load_xdmf('run_b.xmf', stride=4)}
viz3d.compare_geometry(cases, component='fac', slice_panel=True)
```

The first mapping entry is the **reference case**, supplying default
thresholds and automatic magnetic-line seeds. `initial_case='Run B'` changes
only the initially displayed case. For reproducible tracing independent
of the initial diagnostic, supply explicit `seeds` in grid coordinates.
The CLI model comparison does this.

### Comparison contract

- All cases must have exactly identical x/y/z axes with at least three
  nodes per axis. Different grids must be resampled before calling the
  viewer; it does not silently align them.
- Coordinates, units, and preprocessing must agree. Conflicting declared
  `coordinate_system`, `length_unit`, or `field_unit` metadata are rejected.
  Missing declarations are allowed; the caller remains responsible for
  consistency. Neither metadata nor labels convert units.
- `max_points`, `mask`, `delta`, `geometry_delta`, `current_scale`, and display unit
  labels apply to every case. Do not mutate the input arrays while viewing.
- All cases use the same evaluation mode. Without `fields`, FAC uses
  Cartesian grid differences and other diagnostics/traces use the linear
  preview interpolant. With `fields`, all magnetic derivatives and traces
  use the selected callable and the shared explicit difference steps.
- NaN stays blank and validity is specific to each diagnostic. A case with
  no valid results keeps its label and reports that the quantity is
  unavailable. Numerical preparation failures retain the previous scene
  and selected labels, and raise an error.

| State | Dataset selection |
| --- | --- |
| Camera, pan, zoom, projection | Preserved in every panel |
| Slice normal/origin and F4 mode | Preserved |
| Diagnostic | Preserved |
| Colour range | Shared across all cases for that diagnostic |
| Absolute threshold | Shared across cases, remembered per diagnostic |
| Magnetic-line seeds | Fixed physical coordinates |
| Magnetic lines | Retraced through the selected direct or interpolated field |
| Volume, projections, regions, current arrows | Replaced together |

Threshold defaults use the reference case's absolute-value percentile
(`percentile=90`). The slider then sets an absolute cutoff; it is never
lowered automatically for a weaker case. Thresholds affect regions,
arrows, and peak maps. Slices show all finite strengths.

The automatic colour limit is the largest per-case 98th percentile of
finite absolute values. If that is zero, it falls back to the global peak
or 1 for entirely zero/invalid data. Colours are symmetric about zero;
nonnegative `gamma` occupies the positive half. Percentile limits can
saturate extremes. `color_limits={'alpha': 0.2, 'fac': 0.05}` overrides
limits in each diagnostic's displayed units. Legends identify the shared
scale. Eta defaults to a fixed [−1, 1] range instead of percentile scaling.
Different diagnostics still have separate scales and units.

### Preparation, memory, and numerical resolution

The first use of each diagnostic makes a synchronous pass over all cases
to establish its common colour limit, slider bounds, and reference
threshold. The CLI announces initial preparation; later diagnostic
switches show progress in the window. Even explicit colour limits require
this pass to set the remaining shared controls. Selection is synchronous
and can pause the interface while computing or tracing.

`cache_size=2` bounds retained case previews and their derived results.
Revisiting an evicted case recomputes it. Small scale summaries remain
cached. During preparation the current scene/selection remains available
until the replacement is ready. Input grids are all held in memory
separately; this is not a streaming file loader or a total process memory
cap. The six default magnetic arrays occupy about 21.4 MiB; derived arrays,
VTK meshes and tracing require additional memory. One scene is reused.

See [resolution and derivatives](#resolution-memory-and-derivatives) for
preview coarsening and step convergence. There is no automatic grid alignment,
streaming file loader, side-by-side case display, or difference-map mode.

### Comparison screenshots

These standalone-viewer images show total-field `alpha`, using direct
T96 + dipole evaluation with a 0.002 Re difference step. They use the
default six-case comparison and 59 × 44 × 44 preview, a YZ slice at
x = −6 Re, and one automatic colour range shared across all six cases.
They show different cases from the same comparison conditions; they are
separate from the Qt guide's coarse residual-gradient eta captures.

| IMF By = -5 nT | IMF By = +5 nT |
| --- | --- |
| ![Standalone alpha slice at x = -6 Re for IMF By = -5 nT](images/comparison-by-negative.png) | ![Standalone alpha slice at x = -6 Re for IMF By = +5 nT](images/comparison-by-positive.png) |

Regenerate from the repository root:

```bash
python -m mageometry.viz3d --by -5 -3 -1 1 3 5 --slice-only --screenshot docs/images/comparison-by-negative.png
python -m mageometry.viz3d --by -5 -3 -1 1 3 5 --initial-by 5 --slice-only --screenshot docs/images/comparison-by-positive.png
```

## Free slice plane and Frenet frames

Use `slice_view(mode='plane')` and `add_frenet_frame` to compose a curvature
slice with tangent, normal, and binormal arrows. This standalone example
uses a synthetic helical field and needs only `.[viz3d]`:

```python
import numpy as np
from mageometry import GriddedField, viz3d

def field(x, y, z):
    x, y, z = np.broadcast_arrays(x, y, z)
    return -y, x, np.ones_like(z)

axes = (np.linspace(0.5, 2, 17), np.linspace(-1, 1, 17),
        np.linspace(-1, 1, 17))
coords = np.meshgrid(*axes, indexing='ij')
grid = GriddedField(*axes, *field(*coords))
plotter = viz3d.slice_view(grid, 'curvature', mode='plane', normal='y',
                          field=field, delta=1e-3, show=False)
viz3d.add_frenet_frame(plotter, field, [0.75, 1.25, 1.75], [0, 0, 0],
                       [0, 0, 0], delta=1e-3, length=0.3)
plotter.show()
```

Drag the plane's arrow to translate the slice and grab the plane to rotate
it. The companion panel follows the plane normal and shows the slice
face-on. Frenet arrows stay at their specified positions in the main 3D
view: T is red, n green, and b blue. Their length is in coordinate units;
the curvature colour scale has inverse coordinate-length units.

Pass `front_view=False` to `slice_view` for a single panel. The returned
plotter leaves the main 3D panel active, so additional `add_*` calls draw
there. To use your data, replace the synthetic grid with a loaded
`GriddedField`, use `field = grid.field()`, and choose arrow locations and
`delta` with room for derivative stencils inside the grid.

## Regenerate the README screenshots

This section covers the standalone PyVista captures. The Qt workspace has
its own [capture script and settings](gui.md#screenshots-and-regeneration).
See the [image index](images/README.md) for the complete file-to-source mapping.

From the repository root with `.[viz3d]` installed, run:

```bash
python benchmark/readme_screenshots.py
# Inspect a separate set before replacing documentation assets:
python benchmark/readme_screenshots.py --output-dir /tmp/mageometry-screenshots
```

The script renders ten PNGs directly from the current viewer using
the shared `mageometry.session.presets.model_snapshot` T96 + dipole data
and source metadata. It
uses the default 120,000-node preview budget, a 1920 × 960 window, the
companion slice panel, and a YZ plane at x = −6 Re. The screenshots in
the README were captured on 2026-09-25 with PyVista 0.49.0.

The capture sequence shows FAC with and without the draggable plane,
the enlarged FAC slice, normal-current overview, binormal-current slice,
first parallel-current term, signed shear difference, open component menu,
and eta in overview and enlarged modes. Current components are selected
through the actual dropdown, retaining the slice position and magnetic
lines. Eta starts in a fresh viewer with its own automatic context seeds.
The strength threshold and per-component colour limits are automatic;
eta retains its fixed [−1, 1] scale. Inspect the resulting images after
changes to layout, labels, or numerical sampling.

The analytic `beta_g` image and IMF By comparison images illustrate separate
examples; the latter have [their own regeneration commands](#comparison-screenshots).
The README's accuracy plots are generated by
[`benchmark/readme_validation.py`](../benchmark/readme_validation.py).

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Missing magnetic arrays | Names and association; the CLI uses the defaults above |
| Unsupported topology | Convert/resample to a rectilinear grid, or write a reader |
| Fewer than three nodes per axis | Reduce stride or enlarge the selected region |
| Blank beta_g/delta_g or Frenet current | Curvature normal and neighbouring stencils; compare `gamma`, `alpha`, or `fac` |
| No regions above the threshold | Lower the threshold and inspect the slice; also check finite values |
| Out-of-domain error in custom Python code | Keep the interpolator's default NaN fill rather than `fill_value=None` |
| Display or OpenGL error | Check the available VTK rendering backend; off-screen rendering still requires a working backend |


## Background contribution comparison

`viz3d.transverse_contribution_view(total_grid, background, component='eta')`
opens with a selected background and adds a contribution dropdown (F7/F8) for total, background-gradient and
residual-gradient diagnostics. A single total-field frame, normalization and
set of magnetic lines is retained. Colour limits and per-diagnostic thresholds
are shared; all six transverse diagnostics are available through F5/F6.

```bash
python -m mageometry.viz3d --background dipole --contribution residual --component gamma --slice x
python -m mageometry.viz3d --vtk total.vti --background-vtk background.vti --component eta
```

For selection in the standalone viewer, start `python -m mageometry.viz3d` and use
**BACKGROUND → Dipole**, followed by **CONTRIBUTION → Residual gradient**.
The ordinary `geometry_view` also has this menu; Python callers register
presets via `background_choices={'Reference': background}`. **None (total
field)** restores the full diagnostic menu. A current diagnostic switches to
eta when a background is enabled.

**Load background file...** opens a folder browser within the same menu,
including parent, home/root, paging and Cancel. Escape or clicking another
menu cancels without changing data. Supported files are `.xmf`, `.xdmf`,
`.vti`, `.vtr`; simulation CLIs apply their total-input stride to backgrounds.
Python callers can configure `background_loader(path)` and
`background_directory`. Invalid files or incompatible grids display a
message while preserving the scene. Changing background recomputes shared
limits while retaining total magnetic lines, cameras, slices and thresholds.

Background grids must have matching axes, coordinates and units; callable
backgrounds are accepted by the Python API. The built-in dipole choice is
only available for the model demo. Contribution eta and omega_c describe
projected gradient operators, not residual-field winding. See the
[background attribution guide](transverse_geometry.md#background-gradient-contributions)
for the API, validity rules, shared-scale policy and interpretation.

Regenerate the background-menu image and six contribution comparison captures with
`python benchmark/transverse_contribution_screenshots.py`. The script selects the dipole background and switches
contributions through actual dropdowns while retaining one total-field trace, camera and x = −6 Re
slice. Inspect the generated images in `docs/images/contribution-*.png`.
