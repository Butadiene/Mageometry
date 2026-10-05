# Desktop and session architecture

[Documentation index](README.md) · [Desktop user guide](gui.md)
· [Scientific baseline](fac_anisotropy_theory.md)

This describes the implemented desktop, not a proposed design. User-facing
controls, numerical options and persistence rules live in the desktop guide;
the standalone Python comparison contract lives in the
[viewer guide](viewer.md#comparison-contract).

## Ownership and dependencies

| Module | Responsibility |
| --- | --- |
| `geometry/` | Callable-field scientific API; no GUI or session dependency |
| `session/specs.py` | Validate/copy JSON recipes and migrate schema defaults |
| `session/sources.py`, `presets.py` | Model/reader construction, epoch binding, masks, input identity |
| `session/engine.py` | Prepare case/contribution arrays, common statistics and traces |
| `session/jobs.py` | One spawned numerical worker and generation-based cancellation |
| `session/persistence.py` | Atomic JSON writes, relative paths and version records |
| `gui/window.py` | Draft/request/displayed recipes, case management, application layout |
| `gui/forms.py` | Source and analysis editors |
| `gui/display.py` | Display controls and synchronization against a prepared scene |
| `viz3d/scene.py` | VTK actors, slices, cameras and layout; no source loading |

Qt imports occur only when launching the application. Session construction,
scientific APIs and off-screen plot export do not need Qt. The engine uses
the internal `_OverviewData`, `_ContributionData` and diagnostic caches also
used by standalone viewers; the two rendering interfaces remain separate.
These session/scene interfaces are development APIs. `mageometry.geometry`
is the primary public analysis API.

## State and worker results

The window distinguishes editable source/analysis settings, the requested
selection, and the recipe attached to the displayed result. Apply freezes
a validated request. Further edits remain a draft, while the result header
continues to describe the displayed calculation. A failed or cancelled request
leaves the latest successfully displayed field and its labels intact.

`JobRunner` sends recipe copies to a single spawned process. Its generation
counter invalidates superseded requests and late responses. Cancellation is
checked between work units and at every trace field evaluation; an individual
library call may finish first. Trace seeds are integrated in batches of at
most 32 using the existing adaptive integrator and unchanged tolerances.
Model evaluators restore their epochs for each call. Stateful geopack models
are evaluated serially, not concurrently in threads. Only arrays, trace paths
and metadata cross back to Qt; VTK objects stay on the GUI thread.

All group inputs and common colour statistics are still prepared before the
first result. The worker publishes field/slice arrays and resolved seeds
before computing missing traces. A later event carries only paths. The window
commits the field immediately and keeps cancellation and display controls
available; trace completion changes only line actors. Generation tokens reject
late path events as well as late field results. Cancellation/failure after
field publication retains that field and marks its context lines unfinished.

`DisplayPanel` owns display widgets and their signal-suppression state. It
updates the existing `GeometryScene`, without submitting numerical jobs or
reading draft analysis settings. The window owns layout/dock visibility and
forwards plane changes to the panel's extent labels. Widget validation errors
are signalled to the window's status display. With no prepared result, the
panel is disabled.

## Reuse and invalidation

The worker's `calculation_key` includes ordered case descriptions, reference,
analysis settings and source revision. A changed key constructs a new
`SessionEngine` with compatible state from its predecessor:

| Retained state | Reuse conditions |
| --- | --- |
| Source grids, evaluators and fingerprints | Same group/revision, source description and mask radius; file size/mtime and saved hashes are checked before reuse |
| Diagnostic previews and common statistics | Same cases/reference and numerical analysis settings, excluding trace configuration and presentation labels |
| Trace paths | Retained source grid, same evaluation mode, preview budget, mask, effective integration options and resolved seed coordinates |

Thus trace-only edits keep input and diagnostic work, while derivative edits
keep input and compatible paths. Trace disable/enable and automatic seeds
becoming explicit do not discard compatible results. A source revision clears
reuse; other numerical edits invalidate the relevant derived state. Reuse is
in memory in the numerical worker, not a persistent on-disk result cache.

Within one engine, changing case/diagnostic/contribution reuses bounded
preview and trace caches (the trace bound counts case/configuration entries).
Common scale summaries are retained per diagnostic;
attribution includes every case and branch. Display-only changes reuse these
arrays. The preview count is not a total memory cap: input grids, VTK meshes,
attribution results and process-transfer copies also consume RAM.

Within `GeometryScene`, `update_visibility` changes existing actors and the
3D plane widget; `update_colors` changes mapper/legend ranges, including a
manual slice override. Neither operation rebuilds geometry, changes cameras
or submits numerical work. Each requests one render. Threshold, interval and
sign edits still use `update_display` to rebuild their filtered geometry;
plane position/orientation and extent edits use `update_slice`.

Automatic seeds come from the reference case and are fixed once resolved.
The window saves their physical coordinates for reproducibility. Trace-disabled
recipes skip seed resolution and integration while retaining the user's seed
and integration settings. Background attribution always uses the total-field
frame and total-field tracing; its mathematical contract is in the
[theory](fac_anisotropy_theory.md#background-attribution-in-the-total-field-frame).

## Persistence and validation

Save/export uses committed recipes plus their current view. A field published
before its traces is already committed. Saving it retains requested trace
settings and resolved seeds; reopening completes tracing again. Cancelled or
failed paths are not represented as a completed trace cache. Recipes contain
no pickled callables, arrays or VTK actors. File identities include SHA-256,
size and timestamps (including XDMF heavy data); cached inputs also check
size/mtime. Reload explicitly accepts a new input revision. Packaged coefficient
assets are not separately fingerprinted. See
[save and restore](gui.md#save-restore-and-export) for schema migration and
portable path behavior.

Run `python -m unittest discover tests/`. Focused controller regressions are
in `test_geometry_gui.py`; scene, worker, session and model tests cover their
respective boundaries. GUI tests substitute a deterministic runner and use
an off-screen VTK scene. For actual Qt mouse/keyboard interaction and screenshots,
run `benchmark/render_gui.py` on a working desktop/OpenGL display; capture
conditions are in the [image index](images/README.md#qt-desktop-workspace).
`benchmark/profile_gui.py` records the actual Qt OpenGL backend and measures
prepared-array display operations; see [rendering performance](gui.md#rendering-performance-and-wsl).
