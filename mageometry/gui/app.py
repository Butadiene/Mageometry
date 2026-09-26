"""Application entry point; Qt is optional until launch."""

import argparse
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


def main():
    parser = argparse.ArgumentParser(description='Unified magnetic geometry workspace')
    parser.add_argument('--session', help='Open a saved JSON session')
    parser.add_argument('--by', type=float, nargs='+', help='Start a T96 By comparison')
    parser.add_argument('--empty', action='store_true', help='Start with an empty file group')
    args = parser.parse_args()
    from ..session import load_session, model_session, empty_session
    if sum(bool(value) for value in (args.session, args.by, args.empty)) > 1:
        parser.error('Choose one of --session, --by, or --empty.')
    session = load_session(args.session) if args.session else empty_session() if args.empty else model_session(args.by or (0.,))
    return run(session)
