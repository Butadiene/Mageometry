# Unified magnetic-geometry application: design proposal

[Documentation index](README.md) · [Interactive layout mockup](gui_mockup.html)
· [Current viewer](viewer.md) · [Baseline theory](fac_anisotropy_theory.md)

Status: accepted layout and design reference. The desktop implementation is
now available; see the [workspace guide](gui.md) for shipped behavior and
current limitations. Some module boundaries and optimizations below remain
design targets. The self-contained HTML mockup demonstrates selection and
editing behavior; it computes no magnetic data and remains a review artifact.

## Purpose and design principles

Make the complete workflows in `fac_viewer.py`, `geometry_viewer.py`,
`geometry_viewer_simulation.py`, and `compare_t96_by.py` accessible from one
desktop application. Users should be able to create or load cases, choose
analysis conditions, compare results, and save a reproducible session without
editing a script or restarting the application.

The organizing concept is an **analysis session**. The four example scripts
provide useful starting presets for that session. Their names need not become
separate application modes. A session contains sources, comparison groups,
analysis settings, and views, with explicit relationships between them.

- Scientific functions remain in `mageometry.geometry`; GUI callbacks contain
  no alternative definitions of diagnostics.
- Every displayed result identifies its case, numerical method, diagnostic,
  units, and, when applicable, background and reference frame.
- Comparison holds relevant conditions fixed. Changing the selected case
  must not silently change its colour scale or numerical method.
- Undefined geometry remains blank. A missing value is not a zero result.
- Preserve independent Cartesian FAC and Frenet current estimates, including
  their different numerical methods and validity masks.
- Save the calculation recipe as well as the view. A screenshot alone is
  insufficient to reproduce a numerical comparison.
- Model parameters are inputs; geometric diagnostics are calculated results;
  physical interpretations remain subject to the assumptions in the theory.

The repository's R1/R2 hypotheses do not become automatic classifications in
the GUI. `beta_g` and `delta_g` retain their canonical names throughout.

## Scope and existing implementation

| Area | Existing implementation | Work required for the application |
| --- | --- | --- |
| Single-case viewers | Two wrappers call `fac_viewer.main` with different defaults or required input | One entry point with startup presets |
| Model inputs | Fixed T96 + dipole example and configurable By cases | Editable T96 source form and reusable source factory |
| Simulation input | XDMF, HDF5, VTK readers | File dialogs and reader forms using those readers |
| Diagnostics | 18 entries in `viz3d._current.COMPONENTS` | Shared registry exposed in a grouped selector |
| Dataset comparison | `compare_geometry`, identical axes, shared settings and scales | Persistent case list and comparison-group management |
| Gradient attribution | `transverse_contribution_view`, six transverse diagnostics | Separate case and contribution dimensions |
| Rendering | Shared overview, slices, projections, lines, arrows | Embed and separate scene updates from input widgets |
| Background selection | Presets and XDMF/VTK browser | Per-case assignments; use the same source forms for backgrounds |
| Screenshots | Interactive/off-screen PyVista output | Export the committed result with its settings |
| Session persistence | No current portable session format | Versioned declarative recipe and restore validation |
| Responsive computation | Viewer preparation is synchronous | Job control, cancellation, and stale-result rejection |

The first complete release covers all four scripts, including **dataset
comparison combined with gradient attribution**, session save/restore, and
responsive job status. Implementation milestones below are delivery steps,
not a reduction of that target.

Additional model families, time-series playback, side-by-side scenes,
difference maps, automatic grid alignment, and the other notebook workflows
are extensions. Existing Python access to those capabilities remains intact.
Ordinary case switching is the initial comparison presentation. A parameter
scan is an ordered case collection; its By values are never stored as times
in `FieldSeries`.

## Window and controls

```text
Session: New / Open / Save             Add model / Add files       Export PNG
---------------------------------------------------------------------------
Cases and groups     | Result: case / diagnostic / contribution / units
                     |-----------------------------------------------------
Group: By scan       | 3D overview + face-on slice      | Analysis / Display
  By = -5 [reference]|                                 | Diagnostic
  By = -3            |                                 | Analysis kind
  By = -1            |                                 | Background
  By = +1            |                                 | Contribution
  By = +3            |                                 | Slice / threshold
  By = +5 [selected] |                                 | Colour scale
                     | Signed peak projections         | Details / Numerics
Source properties    |---------------------------------+-------------------
                     | Result conditions / pending changes / job progress
---------------------------------------------------------------------------
                       Apply and recompute / Discard edits / Cancel job
```

Use resizable docks for the case list and settings, leaving the plot usable
on smaller screens. Plot navigation retains existing mouse gestures.

### View layouts and focused inspection

Provide three explicit layout buttons above the result. They change the
allocation of screen space and require no magnetic recomputation or Apply.

| Layout | Visible plot panels | Typical task |
| --- | --- | --- |
| All panels | 3D overview, face-on slice, signed peak projections | Locate a structure and relate it to the surrounding field |
| 3D focus | Enlarged 3D overview, including its optional draggable plane | Inspect spatial structure, field lines, and current arrows |
| Slice focus | Enlarged face-on slice | Compare fine spatial patterns at fixed physical coordinates |

Keep dataset, diagnostic, contribution, result conditions and layout buttons
visible in every layout. Put plane orientation and position beside the plot
so that slice navigation remains accessible when the settings dock is hidden.
Provide **Hide side panels / Show side panels** independently of plot layout.
This hides the case/source dock and settings dock to expand the result across
the window; a compact dataset selector remains above the result. Restore is
always available in that same toolbar. This is separate from operating-system
full-screen mode and does not require it.

The two renderers retain separate view state: the 3D camera and the slice's
face-on pan/zoom. The plane normal/origin is shared. Enlarging a panel keeps
its centre and zoom; restoring the layout does not reset either camera.
The first slice view may fit the data once; subsequent fits are explicit
Reset/Fit actions. Case, diagnostic, contribution, colour limits, thresholds,
trace seeds, draft edits and job state are unchanged by layout switches.
The selected layout also survives case or diagnostic changes and job results.
Hidden panels reuse prepared data when shown again.

F4 toggles Slice focus and the preceding non-slice layout (All panels or
3D focus). With plot navigation focused, Escape restores All panels and the
side docks; if a menu or dialog is open, it handles Escape first. Shortcuts
must not intercept text or numeric editing. Provide visible buttons for all
these actions. Store layout, dock visibility, splitter sizes, and both camera
states in the session. PNG export uses the selected plot layout and includes
case, diagnostic, contribution, units and colour scale, independently of dock
visibility. A separate export-layout override can be added later.

The HTML mockup implements the three layouts, side-panel visibility, compact
dataset selection and plot-focused F4/Escape. It demonstrates state retention
with the plane, selection and draft settings; its schematic SVGs do not
implement actual 3D rotation, camera persistence, or slice pan/zoom.

### Cases and source properties

The case list shows stable labels, source type, availability, the selected
case, and the explicitly marked reference case. The group selector separates
incompatible datasets. Selecting a case does not change the reference case.
Adding, duplicating, renaming, removing, and regrouping cases are explicit
actions; removal from a session does not delete source files.

**Add model** opens a T96 + dipole form: epoch, Pdyn, Dst, IMF By/Bz,
domain bounds, grid shape, and inner exclusion radius. Dipole tilt is derived
from the epoch and shown as a result. This preserves the example's recalc
convention; an independent tilt override is not introduced implicitly.
The By field can be expanded into an explicit list of values using
**Create parameter scan**. The resulting cases have separate identities and
immutable parameter snapshots.

**Add files** chooses XDMF, HDF5, or VTK and opens the corresponding reader
form. HDF5 requires coordinates; XDMF offers an optional heavy-file override.
Show array names, axis order where supported, stride, declared coordinates,
and units before applying. A source edit creates a new revision; it must not
mutate arrays still used by the displayed result. File backgrounds use the
same forms, including direct HDF5 with explicit coordinates.

The default model preset exactly records the examples' conditions: T96 +
dipole, epoch 100 Unix seconds, Pdyn 2 nPa, Dst -20 nT, By 0 nT, Bz -5 nT,
GSM coordinates, Re/nT, x in [-15, 5], y/z in [-8, 8], shape (65, 49, 49),
inner exclusion radius 2.5 Re, and a reference sphere of radius 1 Re.
These are reproducibility defaults, not recommended conditions for every
scientific question. The By-scan preset uses [-5, -3, -1, 1, 3, 5] nT.

### Analysis and diagnostic selection

Keep the following independent: comparison group, case, analysis kind,
background assignment, contribution, and diagnostic.

| Analysis kind | Available contributions | Available diagnostics |
| --- | --- | --- |
| Field geometry and currents | Total field | All 18 current viewer diagnostics |
| Gradient attribution in total-field frame | Total, background gradient, residual gradient | `alpha`, `beta_g`, `delta_g`, `gamma`, `omega_c`, `eta` |

A background can be assigned without changing the current analysis kind.
Choosing attribution requires a valid assignment for every participating
case. Show the current case's assigned background beside the selector.
The model preset can assign a dipole reconstructed at each case's epoch.
Simulation inputs require an explicit background; filenames or metadata
never imply a dipole.

Selecting attribution while a current diagnostic is active presents `eta`
as the proposed diagnostic in the settings, with a short explanation, before
Apply. Ineligible diagnostic entries stay visible but disabled with the
reason. Returning to field analysis restores its last diagnostic. Case and
contribution selectors must never share the same state variable.

Group the diagnostic menu into currents, current terms, and transverse
geometry. Each entry shows its canonical key, description, basis when
relevant, and unit. A help panel links to the baseline theory. In attribution,
label alpha as projected rotation, and state that magnetic context lines
follow the total field for every branch.

### Display and numerical settings

Display contains view layout, slice orientation/origin, threshold, colour
limit, layer visibility, camera alignment/reset, and screenshot export.
An x-normal slice is labelled **YZ (normal: x)**. Threshold affects regions,
arrows, and peak-map visibility; slices retain all finite values. Peak maps
are labelled signed maximum-absolute projections, not integrated currents.

Numerics contains direct/grid evaluation, Cartesian FAC step, geometry
step, preview node budget, exclusion mask, trace seeds/options, and cache
capacity. Show both requested input shape and resolved preview shape.
In grid mode, label the budget **Analysis/display preview nodes**, since
coarsening changes the interpolated field used by derivatives and tracing.
The FAC step is read-only as **Preview axis spacing** in that mode.

Direct mode uses an explicit positive FAC step (preset 0.002 Re); the
geometry step defaults to its minimum and can be overridden independently.
The effective steps are displayed even when the settings use an automatic
default. Direct evaluation still uses finite differences. In either mode,
the rendered slices interpolate sampled diagnostics on the preview grid.

The model current conversion is explicitly recorded as scale 0.125 from
native nT/Re to approximately nA/m². Transverse rates keep inverse-length
units, and eta is dimensionless. Metadata edits and unit labels do not
convert numerical data. The planetary sphere and numerical exclusion mask
are separate settings.

### Existing CLI coverage

| Existing option or control | GUI location |
| --- | --- |
| `--xmf`, `--vtk`, `--h5` | Add files / source type / XDMF heavy-file override |
| `--origin`, `--spacing`, `--stride` | Reader form, including required HDF5 coordinates |
| `--by`, `--initial-by`, `--shape` | Model scan, selected case, model grid |
| `--evaluation`, `--delta`, `--geometry-delta` | Numerics and resolved-step readout |
| `--max-points`, `--cache-size` | Numerics / resources |
| `--component` | Diagnostic selector; all 18 canonical keys |
| `--background`, `--background-xmf`, `--background-vtk` | Background assignment |
| `--contribution` | Analysis kind / contribution selector |
| `--threshold`, `--color-limit` | Display, with shared scale scope |
| `--slice`, `--slice-origin`, `--slice-only` | Plane controls and enlarged slice |
| `--screenshot` | Export committed view to PNG |
| F1/F2/F3, F4, `c` | YZ/XZ/XY, Slice focus toggle, main-view slice toggle |
| F5/F6, F7/F8 | Previous/next diagnostic; previous/next dataset |
| Contribution previous/next | Visible buttons and Alt+Left/Alt+Right in the new GUI |
| `x`/`y`/`z`, `r`, `l`/`a`/`s` | Camera axis/reset, line/arrow/region toggles |
| Shift+drag, wheel | Pan and zoom in the plot |

Keyboard shortcuts operate when plot navigation has focus and do not
intercept typing in forms. Legacy viewer shortcuts retain their present
meaning; the new GUI always reserves F7/F8 for datasets.

## Three complete user workflows

### A. Explore one model and separate background contributions

1. Choose **New > T96 snapshot**. Review the visible model preset and Apply.
2. Select `fac` or a current component, orient the slice, and adjust its
   position and threshold. The displayed conditions identify the result.
3. Assign **Dipole at case epoch**, choose **Gradient attribution**, and
   choose `eta` or `gamma`. Apply to prepare the three branches.
4. Switch between total/background/residual. The plane, camera, total-field
   lines, common scale, and threshold stay fixed.
   Alternate All panels, 3D focus, and Slice focus as needed; hide side panels
   for more plot space and continue selecting cases/diagnostics above the plot.
5. Save the session and export the displayed result.

### B. Compare By cases and their residual gradients

1. Choose **New > T96 By comparison**. Edit the By list if needed and Apply.
2. Keep the first case as the marked reference, or explicitly choose another
   before preparation. Selecting the initial view never changes it.
3. Choose YZ at x = -6 Re, with the same six seeds as the current comparison
   example: x in {-6, -10}, y in {-2, 0, 2}, z = 1 Re.
4. Assign the per-case dipole preset to the group, then choose attribution,
   residual gradient, and `eta`. Apply.
5. Select By = -5 and +5. Compare under a fixed [-1, 1] eta scale. Change to
   `gamma`; wait for the common scale to be prepared, then repeat selection.
6. Change geometry step from 0.002 to 0.001 Re and Apply for a convergence
   check. The previous result remains labelled with its original step until
   the replacement succeeds. Save the session under a distinct name if both
   analysis recipes should be retained.

### C. Compare simulation snapshots

1. Add two XDMF/VTK snapshots, or HDF5 files with coordinates. Set reader
   stride and the declared coordinate/field units.
2. Add them to one comparison group after checking identical loaded axes,
   compatible declarations, and common preprocessing/evaluation settings.
   Incompatible cases can remain in separate groups. No silent resampling
   or unit conversion is performed.
3. Choose a diagnostic, a plane, and Apply. The comparison uses grid
   evaluation; changing preview budget is a numerical change.
4. If attribution is needed, assign a matching background to each case and
   Apply in attribution mode. A missing/incompatible background identifies
   the affected case and leaves the prior result visible.
5. Save the recipe. Reopening resolves file paths and checks source identity
   before reconstructing results; missing files are shown for relocation.

The mockup provides these three starting contexts, independent selectors,
pending settings, Apply/Cancel, slice controls, three plot layouts, optional
side-panel hiding, and a demo-state export.
Source dialogs, magnetic rendering, full job/cache behavior, and production
session persistence are described here rather than implemented in the mockup.

## Scientific comparison contract

The initial group contract follows `compare_geometry`: exactly matching
x/y/z axes with at least three nodes per axis, compatible declared
coordinates and units, one evaluation mode, common derivative settings,
mask policy, preview budget, and current conversion. Missing metadata is
shown as unspecified; the user can declare it in source properties. Labels
never establish a numerical conversion or prove undeclared compatibility.

A session may hold multiple groups with separate view/settings state.
Group changes restore that group's plane and camera rather than applying
out-of-domain coordinates from an unrelated dataset. Within a group, case
changes preserve the camera and plane and retrace through the selected
total field from fixed physical seed coordinates. Save resolved automatic
seeds so that restore and diagnostic switching cannot change them.

In field analysis, a diagnostic's automatic colour limit is the largest
per-case 98th percentile of finite absolute values. In attribution, extend
the scope to **all cases and all three branches** in the group. Thus both
selectors preserve the same scale. Eta defaults to [-1, 1]; gamma uses the
positive half of the existing diverging palette. Retain the existing peak/1
fallback for zero/empty data. Manual limits have the same scope.

Initial thresholds use the reference case, total branch, and the existing
90th-percentile default. Store thresholds per group, analysis kind, and
diagnostic, shared across cases/branches. Background changes invalidate
automatic attribution statistics but retain explicit thresholds and manual
limits when their units are unchanged. Group-membership changes invalidate
shared statistics. Do not display a newly included case under an old
automatic scale labelled as covering the new group.

Attribution keeps the total-field direction, projection, magnitude, and
Frenet frame where defined. Its residual is formed from gradient tensors
before constructing scalar diagnostics. In particular,
`gamma_residual != gamma_total - gamma_background` in general, and the same
restriction applies to eta. A zero residual gradient has undefined eta.
Only `beta_g` and `delta_g` among the six transverse diagnostics require a
defined curvature normal. Diagnostics of a separately constructed residual
magnetic field answer a different question and are not substituted here.

## State, jobs, and cache invalidation

Maintain editable **draft settings**, the **committed result recipe**, and
the **requested selection** separately. The result header always reflects
the committed recipe. Pending or failed work cannot relabel the old result.
Apply freezes a recipe revision; subsequent edits remain a new draft.

| Change | Required work | Preserved state |
| --- | --- | --- |
| Plot layout or dock visibility | Resize/show existing panels | Both cameras, plane, all analysis/selection/draft/job state |
| Camera, layers, threshold, slice | Update actors or slice cached volume | Magnetic evaluation, derivatives, traces |
| Manual colour limit | Update display scale | Native diagnostic arrays and traces |
| Case selection | Prepare missing case/diagnostic; trace if uncached | Shared comparison scale, plane, camera, threshold |
| Contribution selection | Use or prepare branch arrays | Total-field traces, plane, camera, common scale |
| Diagnostic selection | Compute missing family and shared statistics | Total-field traces, plane, camera; restore diagnostic threshold |
| Background | Recompute attribution and its statistics | Total-field derivatives/traces, camera, slice |
| Geometry step | Recompute geometry/current families and attribution | Independent FAC and traces if FAC step/field unchanged |
| Direct FAC step | Recompute Cartesian FAC; geometry if its step follows the default | Traces and source grid |
| Source parameters, grid, stride, mask, preview budget, evaluation mode | Invalidate dependent arrays, statistics and affected traces | Valid view preferences; explicit source revision |
| Seeds or trace options | Retrace | Diagnostic arrays and scales |
| Cache capacity | Evict unused cached results | Numerical meaning and committed scene |

Selection starts missing preparation automatically; expensive edits wait for
Apply. While a request runs, show its target and progress separately from
the committed result. Cancel discards that request without clearing the
view. Cancellation is cooperative between work units; a running numerical
call may finish before it is acknowledged. New requests supersede queued
older requests. A generation/revision token prevents late results from
overwriting a newer selection. Failure retains the prior scene and gives
an actionable error including the affected source or case.

Use one dedicated numerical worker process initially. Reconstruct model
callables inside it from immutable source descriptions; re-establish the
epoch before evaluating each case or its background. Do not send the
example's closures through a process queue or evaluate stateful geopack
models concurrently in threads. Keep VTK scene construction and mutation
on the GUI thread. Worker results carry NumPy arrays and trace paths, not
live plotters. A pure in-process executor supports deterministic headless
tests without changing calculation semantics.

Cache keys include source revision, resolved numerical settings, mask,
analysis kind, background revision, branch, and diagnostic family. Trace
keys separately include the total field, validity domain, seeds and trace
options. Statistics keys include ordered group membership, reference case,
scale scope, diagnostic, and conversion. Rendering-only state is not part
of a native-array cache key.

Initially keep the existing bounded preview-cache model; source arrays,
display arrays and process-transfer copies consume additional memory.
Show the preview-cache capacity separately from source memory. This is not
a hard process memory cap. Streaming long series and shared-memory transfer
optimizations are subsequent engineering work.

## Software boundaries and migration

Proposed modules (names are provisional; they do not exist yet):

```text
mageometry/session/
    specs.py           # Immutable source, group, analysis and view descriptions
    sources.py         # Registered model/reader factories and background binding
    controller.py      # Draft/commit, selection, validation, invalidation
    jobs.py            # Headless executor and process job messages
    persistence.py     # Versioned session recipes and input fingerprints
mageometry/gui/
    app.py             # Application/event-loop ownership
    window.py          # Docks, menus, progress and status
    forms.py           # Source, analysis and display editors
mageometry/viz3d/
    scene.py           # Extracted scene creation/update API, no source loading
```

`CaseSpec` has a stable ID, label, source description, and background binding.
`ComparisonGroupSpec` owns ordered case IDs, reference ID, common analysis
settings and comparison scales. `ViewState` owns selection, planes, separate
3D/slice cameras, layout, dock visibility, splitter sizes, layers and
per-diagnostic thresholds. A prepared result records resolved
settings and input revisions. Labels are editable presentation, not cache
keys. Arbitrary Python callables remain usable in existing APIs; a portable
GUI session initially supports the registered model/reader factories.

Reuse `_CurrentPreview`, `_OverviewData`, and `_ContributionData` numerical
logic through extraction, preserving regression coverage. In particular,
replace the internal reuse of `case` for contribution labels with separate
dimensions. Generalize their shared statistics without changing scientific
definitions. Extract the rendering callbacks in `fac._overview_view` into
a controller/scene boundary; adding more parameters to its nested callbacks
would make state ownership harder to verify.

The proposed desktop shell is **PySide6 + PyVistaQt**, retaining PyVista/VTK
rendering. PyVistaQt documents `QtInteractor` embedding and a PySide6 binding
configuration in its [official usage guide](https://qt.pyvista.org/usage.html).
The application owns one event loop; numerical responsiveness comes from
the job design above. Embedding alone does not make calculations asynchronous.

The current viewer rejects its companion-panel layout with an externally
supplied plotter. Therefore embedding requires the explicit scene/layout
extraction; merely passing a `QtInteractor` into the current call is not the
integration plan. Use Qt widgets for forms, case management and file dialogs,
and retain VTK interaction for plots and the draggable slice.

GUI dependencies belong in a new optional extra and must be imported only
when launching the application. Keep Python 3.9+ compatibility, existing
extras, offline package import, and current CLI/Python entry points. Resolve
and test a compatible Qt/PyVista dependency set on Python 3.9 and a current
Python before fixing dependency bounds; newest dependency releases are not
assumed to support every project Python version. The dependency matrix is an
implementation check, not a user design decision.

Move reusable model/reader construction out of `examples/` into the source
factories. Examples can then become thin recipes. Standalone viewer adapters
continue returning `pyvista.Plotter` with their documented defaults, errors,
and `show=False` behavior. CLI and GUI route equivalent settings through the
same preparation layer, retaining off-screen image generation.

## Session persistence and export

Use a versioned JSON recipe with distinct sections for sources, cases,
groups, analysis, views, and provenance. Store source descriptors rather
than pickled callables. Keep original and resolved defaults: effective
derivative steps, preview axes/indices, seed coordinates, automatic scales
and thresholds, and application/dependency versions.

File descriptors include paths relative to the session when possible,
reader options, dataset names/order, declared units and coordinates, and
source identity records. XDMF identity includes its referenced heavy data
or explicit override. Record size/mtime as an inexpensive change detector
and content hashes for verified input identity; hash large inputs in a
cancellable background task. Unverified identity is recorded explicitly.
Never claim that a path or matching timestamp guarantees identical input.
Generated model sources record all inputs and coefficient identity where
applicable. Model parameters and provenance must come from the same frozen
description used to build the evaluator.

Source arrays, VTK actors, and caches are not embedded in the JSON. Relocating
a file preserves its logical source ID only after validation and updates its
recorded revision when content changed. Unsupported schema versions or
unavailable factories produce clear restore errors without executing code
from the recipe. Unknown optional metadata can be retained through migration.

Save defaults to the committed recipe and current view. If unapplied edits
exist, clearly label the action **Save displayed session**; optionally save
draft settings in a separate section marked unapplied. Restore never treats
that draft as the displayed result. Cached arrays are reconstructed from
inputs, and changed inputs or software versions are identified; numerical
and pixel-perfect equality across versions is not promised.

Export PNG captures the committed view and can write an accompanying recipe.
It must not attach pending settings to the old image. Off-screen rendering
uses the same prepared-result and scene interfaces. The HTML mockup's
demo-state JSON is explicitly a separate format, not this session schema.

## Implementation milestones and acceptance criteria

1. **Extract session and scene boundaries.** Preserve the existing viewers
   and numerical regression suite. Verify headless preparation and isolated
   model epochs; introduce no Qt requirement for current imports.
2. **Deliver the desktop shell with existing feature coverage.** All four
   example presets, reader forms, diagnostics, numerical/display controls,
   case switching, attribution, screenshot export, and job status work from
   the application. Keep the former display visible on failure or cancel.
3. **Complete the unified workflow.** Combine datasets and contributions,
   implement versioned save/restore, record resolved defaults and provenance,
   and finish the three end-to-end workflows above.

Milestones 1 and 2 are intermediate states. The unified application is ready
only when milestone 3 and the following checks pass:

- Compare all 18 diagnostics with equivalent existing API calls on a small
  synthetic field, preserving undefined masks and unit behavior.
- Check three-branch decomposition against
  `field_line_transverse_decomposition`, including zero residual gradients
  and undefined curvature frames; never test scalar subtraction as attribution.
- Alternate cases and contributions with identical cameras, planes, scales,
  thresholds, and resolved seeds. Check direct/grid step behavior separately.
- Change model epoch and evaluate another case, then return to the first;
  verify unchanged results and matching source metadata.
- Complete a superseded job late, cancel a job, and inject a reader or
  numerical failure. None may replace or relabel the committed scene.
- Save/restore both model and file sessions, relocate files, detect changed
  inputs, and verify recipes with pending edits preserve committed conditions.
- Exercise every CLI-coverage row through GUI controls. Switch all three
  layouts with a nondefault plane, camera, zoom, case, contribution and
  threshold; retain those states without starting numerical work. Repeat
  while a draft or job exists, and after a new result commits. Check F4's
  return layout, Escape precedence, text-entry focus, hidden-dock access,
  smaller-window behavior, session restore and PNG/recipe consistency.
- Run `python -m unittest discover tests/` for the implementation, with
  focused `unittest` regressions for new state and numerical behavior and
  optional-dependency skips for GUI checks. Add GUI screenshots to its guide.

The review artifact is the mockup and these workflows. User feedback can
focus on the order and accessibility of research tasks; module boundaries,
worker mechanics, and compatibility checks are implementation responsibilities.
