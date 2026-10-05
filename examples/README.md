# Examples and viewer entry points

[Project README](../README.md) · [Desktop workspace](../docs/gui.md)
· [Standalone viewer](../docs/viewer.md) · [Notebooks](notebooks/README.md)

Run these commands after an editable install from the repository root.

## Choose a viewer

| Workflow | Entry point | Install extra |
| --- | --- | --- |
| Edit model/file sources, compare datasets and gradient contributions, save sessions | `python -m mageometry.gui` | `.[gui]` |
| Select a source and diagnostic from the terminal, inspect in PyVista, or export an off-screen PNG | `python -m mageometry.viz3d` | `.[viz3d]`; add `io` for XDMF/HDF5 |
| Build a viewer from your own Python data | `mageometry.viz3d.geometry_view`, `compare_geometry`, `transverse_contribution_view` | `.[viz3d]` |

`geometry_gui.py` launches the desktop workspace. Installed commands are
`mageometry-gui` for the desktop and `mageometry-viewer` for the standalone
viewer. The two interfaces share model source evaluation and
analysis code; their controls and screenshots are specific to each interface.

```bash
python -m mageometry.gui
python -m mageometry.gui --by -10 -5 0 5 10 --layout three_d_slice
python -m mageometry.gui --empty

python -m mageometry.viz3d --component gamma
python -m mageometry.viz3d --xmf snapshot.xmf --stride 4
python -m mageometry.viz3d --by -10 -5 0 5 10 --component alpha
python -m mageometry.viz3d --background dipole --component eta --contribution residual
python -m mageometry.viz3d --component fac --slice x --slice-origin -6 0 0 --slice-only --screenshot fac.png
```

Replace file paths with your own. Run `python -m mageometry.viz3d --help`
for reader, derivative, grid-resolution and display options. Use the desktop
workspace for combined dataset/background-contribution comparison.

## Desktop example launchers

Three scripts provide distinct starting points for the same Qt workspace
and require `.[gui]`. They accept the shared source, analysis and display options:

| Launcher | Default behavior |
| --- | --- |
| [geometry_gui.py](geometry_gui.py) | Standard entry point: single T96 + dipole model starting on `alpha` |
| [geometry_viewer_simulation.py](geometry_viewer_simulation.py) | Requires an explicit snapshot file or saved session |
| [compare_t96_by.py](compare_t96_by.py) | Five-case scan at Dst = −30 nT, IMF Bz = −10 nT, By = −10, −5, 0, +5, +10 nT |

They contain no separate viewer logic. For example:

```bash
python examples/compare_t96_by.py --layout three_d_slice
python examples/geometry_gui.py --component fac --layout three_d_slice
python examples/geometry_viewer_simulation.py --xmf snapshot.xmf --stride 4
```

The redundant `geometry_viewer.py` and `fac_viewer.py` launchers have been
removed. Replace the former with `geometry_gui.py` or `python -m mageometry.gui`;
replace the latter with either command plus `--component fac`. FAC is also
available from the GUI's **Diagnostic** selector.

For desktop numerical settings, `--geometry-delta` sets the shared derivative
step; optional `--fac-delta` overrides only the directly computed FAC.

`--screenshot` exports the desktop plot layout plus a matching session recipe.
The standalone interface remains available as `python -m mageometry.viz3d`.

Import model helpers (`model_snapshot`, `make_cases`, `make_fields`) from
`mageometry.session.presets`. Replace imports from the removed `fac_viewer`
example with `from mageometry.session.presets import model_snapshot`.
`compare_t96_by.py` provides `make_cases` and `make_fields` wrappers using
the same five-case scan and Dst/Bz conditions as **New By comparison**, the
desktop/standalone `--by` launchers, and the underlying helpers: Dst = −30 nT,
Bz = −10 nT. Override these with CLI `--dst` / `--bz` or the helpers' `dst` /
`bz` keywords. `comparison_session()` creates this preset as a portable recipe
without evaluating fields. `model_snapshot()` and the standard single-model
desktop retain Dst = −20 nT and Bz = −5 nT.

## Numerical examples and figures

These demonstrate Python analysis and validation beyond launching the GUI:

- [readme_examples.py](readme_examples.py): geometry calculations without
  plotting or input files; requires only the base installation.
- [mhd_gridded_field_example.py](python_code_samples/mhd_gridded_field_example.py):
  dipole-oriented diagnostics for a caller-supplied XDMF/HDF5 snapshot.
- [Notebook index](notebooks/README.md): field geometry, simulation input,
  visualization and current-density derivations.
- [Documentation image sources](../docs/images/README.md): screenshot and
  concept-figure recipes under `benchmark/`, with capture conditions.
