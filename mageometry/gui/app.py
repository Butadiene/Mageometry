"""Application entry point; Qt is optional until launch."""

import os
import sys


def run(session=None):
    os.environ.setdefault('QT_API', 'pyside6')
    try:
        from PySide6.QtWidgets import QApplication
        from .window import MainWindow
    except ImportError as exc:
        raise ImportError("The desktop GUI requires the gui extra: python -m pip install -e '.[gui]'"
                          f'\n{exc}') from exc
    owned = QApplication.instance() is None
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = MainWindow(session)
    window.show()
    if owned:
        return app.exec()
    return window


def main(argv=None, **presets):
    from .cli import main as launch
    return launch(argv, **presets)
