"""Measure Qt display updates without file loading or magnetic-field evaluation.

Run on a working desktop after installing .[gui]:
    python benchmark/profile_gui.py --output /tmp/mageometry-profile

Writes OpenGL capabilities, per-operation timings and two plot captures.
Use the same window size, node count and rendering backend when comparing
code revisions. Timings include synchronous VTK rendering, not desktop
compositor/input latency. No driver or quality settings are changed here.
"""

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import pyvista as pv
import pyvistaqt
from PySide6.QtWidgets import QApplication

from mageometry.gui.window import MainWindow
from mageometry.session.specs import default_analysis, default_view


def synthetic_result():
    """Prepared arrays near the default preview budget, with a masked centre."""
    axis = np.linspace(-5., 5., 49)
    x, y, z = np.meshgrid(axis, axis, axis, indexing='ij')
    values = np.sin(x) * np.cos(y / 2) + .3 * np.sin(z)
    values[x*x + y*y + z*z < 1.] = np.nan
    basis = np.broadcast_to([1., 0., 0.], values.shape + (3,)).copy()
    t = np.linspace(-5., 5., 500)
    paths = [np.column_stack([t, np.sin(t + phase), np.cos(t + phase)])
             for phase in np.linspace(0., 2*np.pi, 10, endpoint=False)]
    return dict(case='synthetic', case_label='Synthetic performance probe',
                component='alpha', contribution='total', kind='field',
                axes=(axis, axis, axis), values=values, basis=basis, paths=paths,
                metadata={}, scale={'limit': 1., 'peak': 1.3, 'threshold': .4},
                label='alpha [1 / grid unit]', analysis=default_analysis())


def measure(operation, repeats):
    for index in range(2):
        operation(index)
    samples = []
    for index in range(repeats):
        start = time.perf_counter()
        operation(index)
        samples.append(1000 * (time.perf_counter() - start))
    return dict(median_ms=statistics.median(samples), min_ms=min(samples),
                max_ms=max(samples), samples_ms=samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=12)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    app = QApplication([])
    window = MainWindow(auto_prepare=False)
    try:
        window.show()
        app.processEvents()
        scene, plotter, panel = window.scene, window.plotter, window.display_panel
        result, view = synthetic_result(), default_view()
        view['origin'] = [0., 0., 0.]
        scene.set_result(result, view)
        panel.sync()
        app.processEvents()
        capabilities = plotter.render_window.ReportCapabilities()
        (args.output / 'opengl.txt').write_text(capabilities)
        report = dict(pyvista=pv.__version__, pyvistaqt=pyvistaqt.__version__,
                      vtk=pv.vtk_version_info, qt_platform=app.platformName(),
                      window_size=list(plotter.window_size), nodes=result['values'].size,
                      opengl=[line for line in capabilities.splitlines()
                              if line.startswith(('OpenGL vendor', 'OpenGL renderer', 'OpenGL version'))],
                      timings={})

        def record(name, operation):
            report['timings'][name] = measure(operation, args.repeats)
            print(f"{name}: {report['timings'][name]['median_ms']:.2f} ms", flush=True)

        def capture(name):
            # Allow VTK's new text actors to settle before comparing pixels.
            plotter.render()
            plotter.render()
            pixels = plotter.screenshot(str(args.output / name), return_img=True)
            if np.ptp(pixels) == 0:
                raise RuntimeError('The rendering backend produced a blank image; timings are not valid.')

        print('\n'.join(report['opengl']), flush=True)
        capture('initial.png')
        cameras = scene.camera_state()

        def rotate(index):
            plotter.renderers[0].camera.Azimuth(1.)
            plotter.render()

        record('camera_render', rotate)
        scene.restore_cameras(cameras)
        record('layer_toggle', lambda i: panel.layers['lines'].setChecked(bool(i % 2)))
        panel.layers['lines'].setChecked(True)
        panel.auto_limit.setChecked(False)
        record('shared_colors', lambda i: panel.color_limit.setValue(1. + .2 * (i % 2)))

        def slice_colors(index):
            if index % 2 == 0:
                panel.slice_color_range.setText('-.2 .8')
            panel.auto_slice_color.setChecked(bool(index % 2))

        record('slice_colors', slice_colors)
        record('threshold', lambda i: panel.set_threshold(.3 + .05 * (i % 3)))
        panel.set_threshold(.4)
        panel.layers['lines'].setChecked(False)
        panel.color_limit.setValue(1.2)
        panel.slice_color_range.setText('-.2 .8')
        panel.auto_slice_color.setChecked(False)
        capture('styled.png')
        if window.last_error:
            raise RuntimeError(window.last_error)
        (args.output / 'timings.json').write_text(json.dumps(report, indent=2) + '\n')
    finally:
        window.close()
        app.processEvents()


if __name__ == '__main__':
    main()
