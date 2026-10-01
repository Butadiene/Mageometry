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
must leave that result and its labels intact.

`JobRunner` sends recipe copies to a single spawned process. Its generation
counter invalidates superseded requests and late responses. Cancellation is
checked between work units; an individual library call may finish first.
Model evaluators restore their epochs for each call. Stateful geopack models
are evaluated serially, not concurrently in threads. Only arrays, trace paths
and metadata cross back to Qt; VTK objects stay on the GUI thread.

`DisplayPanel` owns display widgets and their signal-suppression state. It
updates the existing `GeometryScene`, without submitting numerical jobs or
reading draft analysis settings. The window owns layout/dock visibility and
forwards plane changes to the panel's extent labels. Widget validation errors
are signalled to the window's status display. With no prepared result, the
panel is disabled.

## Reuse and invalidation

The worker's `calculation_key` includes ordered case descriptions, reference,
analysis settings and source revision. If this key changes, it constructs a
new `SessionEngine`. This is conservative: changing a derivative step or trace
settings can rebuild more than the strictly affected family. The implementation
does not promise independent fine-grained reuse for those edits.

Within one engine, changing case/diagnostic/contribution reuses bounded
preview and trace caches. Common scale summaries are retained per diagnostic;
attribution includes every case and branch. Display-only changes reuse these
arrays. The preview count is not a total memory cap: input grids, VTK meshes,
attribution results and process-transfer copies also consume RAM.

Automatic seeds come from the reference case and are fixed once resolved.
The window saves their physical coordinates for reproducibility. Trace-disabled
recipes skip seed resolution and integration while retaining the user's seed
and integration settings. Background attribution always uses the total-field
frame and total-field tracing; its mathematical contract is in the
[theory](fac_anisotropy_theory.md#background-attribution-in-the-total-field-frame).

## Persistence and validation

Save/export uses committed recipes plus their current view. Recipes contain
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
