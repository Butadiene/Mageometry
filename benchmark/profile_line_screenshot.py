"""Exercise desktop line selection, profiling and restore, then capture the UI.

Run with a working Qt/OpenGL display:
    python benchmark/profile_line_screenshot.py --output /tmp/mageometry-profiles
"""

import argparse
from pathlib import Path
import time

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from mageometry.gui.window import MainWindow
from mageometry.session import model_session, save_session, load_session
from mageometry.session.profiles import export_profile


def wait_for(app, window, condition):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        app.processEvents()
        if window.last_error:
            raise RuntimeError(window.last_error)
        if condition():
            return
        time.sleep(.02)
    raise RuntimeError('Timed out: ' + window.status.text() + '; ' + window.profile_panel.info.text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/mageometry-profiles'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    session = model_session((5.,))
    group = session['groups'][0]
    group['cases'][0]['source']['shape'] = [17, 13, 13]
    group['view']['profile'].update(visible=True, quantities=['alpha', 'gamma', 'eta'])
    window = MainWindow(session)
    try:
        window.resize(1700, 1160)
        window.show()
        panel = window.profile_panel
        wait_for(app, window, lambda: panel.data is not None)
        window.plot_splitter.setSizes([430, 440])
        window.plotter.renderers[0].camera.zoom(1.4)
        app.processEvents()
        window.plotter.render()
        window.grab().save(str(args.output / 'profile-before-pick.png'))
        # Pick a different actual VTK line through a Qt right-click.
        record = window.scene.result['trace_records'][2]
        panel.pick.setChecked(True)
        renderer = window.plotter.renderers[0]
        renderer.SetWorldPoint(*record['points'][len(record['points']) // 4], 1.)
        renderer.WorldToDisplay()
        x, y, _ = renderer.GetDisplayPoint()
        target = window.plotter.interactor
        ratio = target.devicePixelRatioF()
        position = QPoint(round(x / ratio), round(target.height() - y / ratio))
        print('Picking', record['seed_id'], 'at', position, 'VTK', window.plotter.window_size,
              'Qt', target.size(), flush=True)
        QTest.mouseClick(target, Qt.MouseButton.RightButton, pos=position)
        app.processEvents()
        window.grab().save(str(args.output / 'profile-after-pick.png'))
        assert panel.lines.currentData() == record['seed_id'], ('Right-click selected', panel.lines.currentData())
        wait_for(app, window, lambda: panel.data is not None and panel.data['seed_id'] == record['seed_id'])
        panel.pick.setChecked(False)
        app.processEvents()
        panel.probe_at(-6.)
        panel.canvas.draw()
        window.grab().save(str(args.output / 'gui-line-profile.png'))
        export_profile(panel.data, args.output / 'line-profile.svg', panel.figure, panel.state)
        export_profile(panel.data, args.output / 'line-profile.csv', view=panel.state)
        original = panel.data
        path = args.output / 'line-profile.session.json'
        save_session(window.saved_recipe(), path)
        window.new_session(load_session(path))
        wait_for(app, window, lambda: panel.data is not None)
        assert panel.data['seed_id'] == original['seed_id']
        for name in original['values']:
            np.testing.assert_allclose(panel.data['values'][name], original['values'][name], equal_nan=True)
        print('Qt right-click selection, worker profiles, export and session restore OK', flush=True)
        print(args.output / 'gui-line-profile.png', flush=True)
    finally:
        window.close()
        app.processEvents()


if __name__ == '__main__':
    main()
