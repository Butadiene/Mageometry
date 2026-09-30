"""Exercise the real Qt workspace and save screenshots of its four layouts.

Run from the repository after installing .[gui]:
    python benchmark/render_gui.py --output /tmp/mageometry-gui

Requires a working desktop/OpenGL display. Each layout gets a full-window PNG,
a plot-only PNG, and a matching .session.json recipe. The final session.json
belongs to the subsequent dataset-switch/Apply/restore check, not the captures.
"""

import argparse
from pathlib import Path
import sys
import time

from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from mageometry.gui.window import MainWindow
from mageometry.session import load_session, model_session, save_session


def check_3d_navigation(window):
    """Send real Qt drags and ensure only the visible 3D camera rotates."""
    scene = window.scene
    assert window.plotter.renderers[0].lights, '3D illumination was lost during preparation'
    cameras = scene.camera_state()
    plane = scene.view['plane']
    layout = scene.view['layout']
    target = window.plotter.interactor
    start = QPoint(int(.15 * target.width()), int(.2 * target.height()))
    end = start + QPoint(40, 30)
    scene.view['plane'] = False
    try:
        for mode in ('three_d_slice', 'slice', 'three_d_slice', 'three_d', 'three_d', 'all'):
            window.set_layout(mode)
            QApplication.processEvents()
            if mode == 'slice':
                continue
            before = scene.camera_state()
            QTest.mousePress(target, Qt.MouseButton.LeftButton, pos=start)
            QTest.mouseMove(target, end, delay=30)
            QTest.mouseRelease(target, Qt.MouseButton.LeftButton, pos=end)
            QApplication.processEvents()
            after = scene.camera_state()
            assert before['0']['position'] != after['0']['position'], f'3D drag failed in {mode}'
            for index in range(1, 5):
                assert before[str(index)] == after[str(index)], f'Hidden camera {index} moved'
    finally:
        scene.view['plane'] = plane
        scene.restore_cameras(cameras)
        window.set_layout(layout)
    print('Qt mouse rotation and layout round trips OK', flush=True)


def check_threshold_controls(window):
    """Exercise numeric entry and slider keys through real Qt input events."""
    tabs = window.settings_dock.widget()
    original = window.threshold.value()
    cameras = window.scene.camera_state()
    tabs.setCurrentIndex(1)
    QApplication.processEvents()
    try:
        window.threshold.setFocus()
        window.threshold.selectAll()
        QTest.keyClicks(window.threshold, '0.25')
        QTest.keyClick(window.threshold, Qt.Key.Key_Return)
        assert window.threshold.value() == .25, 'Manual threshold entry failed'
        window.threshold_slider.setFocus()
        QTest.keyClick(window.threshold_slider, Qt.Key.Key_Right)
        assert window.threshold.value() > .25, 'Threshold slider did not update the numeric input'
        window.auto_threshold_range.setChecked(False)
        window.threshold_slider_upper.setFocus()
        window.threshold_slider_upper.selectAll()
        QTest.keyClicks(window.threshold_slider_upper, '1e-1')
        QTest.keyClick(window.threshold_slider_upper, Qt.Key.Key_Return)
        assert window.threshold_slider_max == .1, 'Manual slider range entry failed'
        assert window.threshold.value() == .1, 'Threshold was not clamped to the smaller range'
        window.threshold_slider.setFocus()
        QTest.keyClick(window.threshold_slider, Qt.Key.Key_Left)
        assert window.threshold.value() < .1, 'Slider did not use the manual range'
        assert not window.busy, 'Threshold input submitted numerical work'
        assert window.scene.camera_state() == cameras, 'Threshold input moved the camera'
    finally:
        window.auto_threshold_range.setChecked(True)
        window.set_threshold(original)
        window.sync_display()
        tabs.setCurrentIndex(0)
    print('Qt numeric threshold entry, manual slider range and slider input OK', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=300.,
                        help='Maximum seconds for captures and restore checks (default: 300)')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    session = model_session((-5., 5.))
    group = session['groups'][0]
    for case in group['cases']:
        case['source']['shape'] = [17, 13, 13]
        case['background'] = dict(kind='dipole', parameters={'epoch': 100.})
    group['analysis'].update(kind='attribution', max_points=3000)
    group['analysis']['trace']['max_steps'] = 50
    group['view'].update(component='eta', contribution='residual')
    app = QApplication(sys.argv[:1])
    window = MainWindow(session)
    window.show()
    started = time.monotonic()
    state = {'index': 0, 'status': 0, 'navigation_checked': False}
    modes = ('all', 'three_d_slice', 'three_d', 'slice')

    def advance():
        if window.last_error or time.monotonic() - started > args.timeout:
            print(window.last_error or 'GUI preparation timed out', flush=True)
            state['status'] = 1
            timer.stop()
            window.close()
            return
        if window.scene.result is None or window.busy:
            return
        if not state['navigation_checked']:
            timer.stop()
            check_3d_navigation(window)
            check_threshold_controls(window)
            state['navigation_checked'] = True
            timer.start()
        if state['index'] < len(modes):
            mode = modes[state['index']]
            window.set_layout(mode)
            app.processEvents()
            window.grab().save(str(args.output / f'workspace-{mode}.png'))
            window.scene.screenshot(args.output / f'plot-{mode}.png')
            save_session(window.saved_recipe(), args.output / f'workspace-{mode}.session.json')
            save_session(window.saved_recipe(), args.output / f'plot-{mode}.session.json')
            if mode == 'three_d_slice':
                window.settings_dock.widget().setCurrentIndex(1)
                app.processEvents()
                window.grab().save(str(args.output / 'workspace-display.png'))
                save_session(window.saved_recipe(), args.output / 'workspace-display.session.json')
                window.settings_dock.widget().setCurrentIndex(0)
            print(f'Rendered {mode}', flush=True)
            state['index'] += 1
        elif state['index'] == len(modes):
            window.cases.setCurrentIndex(1)
            state['index'] += 1
        elif state['index'] == len(modes) + 1:
            if window.scene.result['case'] != group['cases'][1]['id']:
                raise AssertionError('Dataset selection did not reach the scene.')
            window.analysis_form.fields['geometry_delta'].setText('0.001')
            window.apply()
            state['index'] += 1
        elif state['index'] == len(modes) + 2:
            assert window.scene.view['layout'] == 'slice'
            assert window.scene.result['resolved']['geometry_delta'] == .001
            save_session(window.saved_recipe(), args.output / 'session.json')
            window.new_session(load_session(args.output / 'session.json'))
            state['index'] += 1
        else:
            assert window.scene.view['layout'] == 'slice'
            assert window.scene.result['case'] == group['cases'][1]['id']
            assert window.scene.result['resolved']['geometry_delta'] == .001
            assert window.plotter.renderers[0].lights, '3D illumination was lost after restore'
            print('Dataset switching, numerical Apply and session restore OK', flush=True)
            timer.stop()
            window.close()

    def check():
        try:
            advance()
        except Exception as error:
            print(f'GUI check failed: {error}', flush=True)
            state['status'] = 1
            timer.stop()
            window.close()

    timer = QTimer()
    timer.timeout.connect(check)
    timer.start(800)
    app.exec()
    return state['status']


if __name__ == '__main__':
    raise SystemExit(main())
