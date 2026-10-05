"""Exercise the real Qt workspace and capture its layouts and sign filters.

Run from the repository after installing .[gui]:
    python benchmark/render_gui.py --output /tmp/mageometry-gui

Requires a working desktop/OpenGL display. Each layout gets a full-window PNG,
a plot-only PNG, and a matching .session.json recipe. The final session.json
belongs to the subsequent dataset-switch/Apply/restore check, not the captures.
"""

import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time

from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from mageometry.gui.window import MainWindow
from mageometry.gui.forms import SourceDialog
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
    original = window.display_panel.threshold.value()
    cameras = window.scene.camera_state()
    tabs.setCurrentIndex(1)
    QApplication.processEvents()
    try:
        window.display_panel.threshold.setFocus()
        window.display_panel.threshold.selectAll()
        QTest.keyClicks(window.display_panel.threshold, '0.25')
        QTest.keyClick(window.display_panel.threshold, Qt.Key.Key_Return)
        assert window.display_panel.threshold.value() == .25, 'Manual threshold entry failed'
        window.display_panel.threshold_slider.setFocus()
        QTest.keyClick(window.display_panel.threshold_slider, Qt.Key.Key_Right)
        assert window.display_panel.threshold.value() > .25, 'Threshold slider did not update the numeric input'
        window.display_panel.auto_threshold_range.setChecked(False)
        window.display_panel.threshold_slider_upper.setFocus()
        window.display_panel.threshold_slider_upper.selectAll()
        QTest.keyClicks(window.display_panel.threshold_slider_upper, '1e-1')
        QTest.keyClick(window.display_panel.threshold_slider_upper, Qt.Key.Key_Return)
        assert window.display_panel.threshold_slider_max == .1, 'Manual slider range entry failed'
        assert window.display_panel.threshold.value() == .1, 'Threshold was not clamped to the smaller range'
        window.display_panel.threshold_slider.setFocus()
        QTest.keyClick(window.display_panel.threshold_slider, Qt.Key.Key_Left)
        assert window.display_panel.threshold.value() < .1, 'Slider did not use the manual range'
        assert not window.busy, 'Threshold input submitted numerical work'
        assert window.scene.camera_state() == cameras, 'Threshold input moved the camera'
    finally:
        window.display_panel.auto_threshold_range.setChecked(True)
        window.display_panel.set_threshold(original)
        window.sync_display()
        tabs.setCurrentIndex(0)
    print('Qt numeric threshold entry, manual slider range and slider input OK', flush=True)


def check_value_sign_control(window):
    """Select each sign through Qt and retain the prepared result and cameras."""
    tabs = window.settings_dock.widget()
    original = window.display_panel.value_sign.currentIndex()
    result = window.scene.result
    cameras = window.scene.camera_state()
    tabs.setCurrentIndex(1)
    QApplication.processEvents()
    try:
        window.display_panel.value_sign.setFocus()
        QTest.keyClick(window.display_panel.value_sign, Qt.Key.Key_Home)
        for mode, excluded in (('positive', 'negative'), ('negative', 'positive')):
            QTest.keyClick(window.display_panel.value_sign, Qt.Key.Key_Down)
            assert window.scene.view['value_sign'] == mode, 'Value sign selection failed'
            assert excluded not in window.plotter.renderers[0].actors, 'Unselected region remains'
            assert window.saved_recipe()['groups'][0]['view']['value_sign'] == mode
            assert window.scene.result is result and not window.busy, 'Sign selection submitted numerical work'
            assert window.scene.camera_state() == cameras, 'Sign selection moved the camera'
    finally:
        window.display_panel.value_sign.setCurrentIndex(original)
        tabs.setCurrentIndex(0)
    print('Qt positive/negative value selection and saved display state OK', flush=True)


def check_model_source_forms(window, output):
    """Exercise the source selector with Qt keys and capture model-specific inputs."""
    for index, model in enumerate(('t96', 't89', 't01', 't04')):
        dialog = SourceDialog(window)
        dialog.show()
        QApplication.processEvents()
        try:
            dialog.kind.setFocus()
            QTest.keyClick(dialog.kind, Qt.Key.Key_Home)
            for _ in range(index):
                QTest.keyClick(dialog.kind, Qt.Key.Key_Down)
            assert dialog.kind.currentText() == model
            for key, expected in (('iopt', model == 't89'), ('g1', model == 't01'),
                                   ('g2', model == 't01'), ('w', model == 't04'),
                                   ('by', model != 't89')):
                assert dialog.fields[key].isVisible() == expected, (model, key)
            QApplication.processEvents()
            dialog.grab().save(str(output / f'source-{model}.png'))
            dialog.accept_source()
            assert dialog.sources[0]['kind'] == model and not dialog.error.text()
        finally:
            dialog.close()
    print('Qt T89/T96/T01/T04 source selection and parameter forms OK', flush=True)


def check_display_ranges(window, output):
    """Enter signed bounds and slice ranges without starting numerical work."""
    scene = window.scene
    original = deepcopy(scene.view)
    cameras = scene.camera_state()
    result = scene.result
    tabs = window.settings_dock.widget()
    window.set_layout('three_d_slice')
    tabs.setCurrentIndex(1)

    def enter(widget, text):
        tabs.widget(1).ensureWidgetVisible(widget)
        widget.setFocus()
        widget.selectAll()
        QTest.keyClicks(widget, text)
        QTest.keyClick(widget, Qt.Key.Key_Return)
        QApplication.processEvents()

    try:
        window.display_panel.threshold_mode.setCurrentIndex(window.display_panel.threshold_mode.findData('interval'))
        enter(window.display_panel.value_interval, '-0.5 0.5')
        window.display_panel.auto_slice_color.setChecked(False)
        enter(window.display_panel.slice_color_range, '-0.6 0.9')
        window.display_panel.auto_slice_extent.setChecked(False)
        enter(window.display_panel.slice_horizontal, '-5 5')
        enter(window.display_panel.slice_vertical, '-4 4')
        key = result['kind'] + ':' + result['component']
        assert scene.view['value_intervals'][key] == [-.5, .5]
        assert scene.view['slice_color_ranges'][key] == [-.6, .9]
        assert scene.view['slice_extent'] == [-5., 5., -4., 4.]
        assert scene.result is result and not window.busy
        assert scene.camera_state()['0'] == cameras['0']
        # At a normal desktop height all controls remain accessible by scrolling.
        tabs.widget(1).verticalScrollBar().setValue(0)
        QApplication.processEvents()
        assert tabs.widget(1).horizontalScrollBar().maximum() == 0, 'Display labels need horizontal scrolling'
        window.grab().save(str(output / 'workspace-display-ranges.png'))
        save_session(window.saved_recipe(), output / 'workspace-display-ranges.session.json')
    finally:
        scene.view.clear()
        scene.view.update(original)
        scene.update_display()
        scene.restore_cameras(cameras)
        window.set_layout(original['layout'])
        tabs.setCurrentIndex(0)
    print('Qt signed intervals, slice colour limits and spatial extents OK', flush=True)


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
    accept_result = window.accept_result

    def capture_first_field(result):
        accept_result(result)
        if 'field_actors' not in state and result.get('trace_status') == 'pending':
            assert window.display_panel.isEnabled()
            assert not result['paths']
            window.plotter.renderers[0].camera.Azimuth(5.)
            state['field_actors'] = [dict(renderer.actors) for renderer in window.plotter.renderers]
            state['field_cameras'] = window.scene.camera_state()
            window.scene.screenshot(args.output / 'plot-tracing.png')
            save_session(window.saved_recipe(), args.output / 'plot-tracing.session.json')
            print('Field and slices displayed before tracing completed', flush=True)

    window.accept_result = capture_first_field

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
            assert 'field_actors' in state, 'No intermediate field result was displayed'
            assert window.scene.result['trace_status'] == 'ready'
            assert window.scene.result['paths']
            assert window.scene.camera_state() == state['field_cameras']
            for index, renderer in enumerate(window.plotter.renderers):
                for name in ('positive', 'negative', 'slice', 'projection'):
                    if name in state['field_actors'][index]:
                        assert renderer.actors[name] is state['field_actors'][index][name]
            print('Trace completion retained field actors and the adjusted camera', flush=True)
            check_3d_navigation(window)
            check_threshold_controls(window)
            check_value_sign_control(window)
            check_model_source_forms(window, args.output)
            check_display_ranges(window, args.output)
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
                for value_sign in ('positive', 'negative'):
                    window.display_panel.value_sign.setCurrentIndex(window.display_panel.value_sign.findData(value_sign))
                    app.processEvents()
                    stem = f'workspace-display-{value_sign}'
                    window.grab().save(str(args.output / f'{stem}.png'))
                    save_session(window.saved_recipe(), args.output / f'{stem}.session.json')
                window.display_panel.value_sign.setCurrentIndex(window.display_panel.value_sign.findData('both'))
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
            window.analysis_form.trace_enabled.setChecked(False)
            window.apply()
            state['index'] += 1
        elif state['index'] == len(modes) + 2:
            assert window.scene.view['layout'] == 'slice'
            assert window.scene.result['resolved']['geometry_delta'] == .001
            assert not window.scene.result['paths']
            assert window.scene.result['resolved']['seeds'] is None
            assert not window.scene.result['analysis']['trace_enabled']
            window.set_layout('three_d_slice')
            window.settings_dock.widget().setCurrentIndex(0)
            QApplication.processEvents()
            window.grab().save(str(args.output / 'workspace-no-trace.png'))
            save_session(window.saved_recipe(), args.output / 'workspace-no-trace.session.json')
            window.set_layout('slice')
            save_session(window.saved_recipe(), args.output / 'session.json')
            window.new_session(load_session(args.output / 'session.json'))
            state['index'] += 1
        else:
            assert window.scene.view['layout'] == 'slice'
            assert window.scene.result['case'] == group['cases'][1]['id']
            assert window.scene.result['resolved']['geometry_delta'] == .001
            assert window.plotter.renderers[0].lights, '3D illumination was lost after restore'
            assert not window.analysis_form.trace_enabled.isChecked()
            assert not window.scene.result['paths']
            print('Dataset switching, disabled tracing, numerical Apply and session restore OK', flush=True)
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
