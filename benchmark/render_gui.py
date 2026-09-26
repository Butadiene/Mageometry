"""Exercise the real Qt workspace and save screenshots of its three layouts.

Run from the repository after installing .[gui]:
    python benchmark/render_gui.py --output /tmp/mageometry-gui

Requires a working desktop/OpenGL display. Exits after rendering the layouts.
"""

import argparse
from pathlib import Path
import sys
import time

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from mageometry.gui.window import MainWindow
from mageometry.session import model_session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
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
    state = {'index': 0, 'status': 0}
    modes = ('all', 'three_d', 'slice')

    def check():
        if window.last_error or time.monotonic() - started > 90:
            print(window.last_error or 'GUI preparation timed out', flush=True)
            state['status'] = 1
            timer.stop()
            window.close()
            return
        if window.scene.result is None or window.busy:
            return
        if state['index'] < len(modes):
            mode = modes[state['index']]
            window.set_layout(mode)
            app.processEvents()
            window.grab().save(str(args.output / f'workspace-{mode}.png'))
            window.scene.screenshot(args.output / f'plot-{mode}.png')
            print(f'Rendered {mode}', flush=True)
            state['index'] += 1
        elif state['index'] == 3:
            window.cases.setCurrentIndex(1)
            state['index'] = 4
        elif state['index'] == 4:
            if window.scene.result['case'] != group['cases'][1]['id']:
                raise AssertionError('Dataset selection did not reach the scene.')
            window.analysis_form.fields['geometry_delta'].setText('0.001')
            window.apply()
            state['index'] = 5
        elif state['index'] == 5:
            from mageometry.session import save_session
            from mageometry.session import load_session
            assert window.scene.view['layout'] == 'slice'
            assert window.scene.result['resolved']['geometry_delta'] == .001
            save_session(window.saved_recipe(), args.output / 'session.json')
            window.new_session(load_session(args.output / 'session.json'))
            state['index'] = 6
        else:
            assert window.scene.view['layout'] == 'slice'
            assert window.scene.result['case'] == group['cases'][1]['id']
            assert window.scene.result['resolved']['geometry_delta'] == .001
            print('Dataset switching, numerical Apply and session restore OK', flush=True)
            timer.stop()
            window.close()

    timer = QTimer()
    timer.timeout.connect(check)
    timer.start(800)
    app.exec()
    return state['status']


if __name__ == '__main__':
    raise SystemExit(main())
