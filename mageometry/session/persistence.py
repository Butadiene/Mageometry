"""Versioned JSON recipes with portable paths and atomic writes."""

import json
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import tempfile

from .specs import validate_session


def _sources(session):
    for group in session['groups']:
        for case in group['cases']:
            yield case['source']
            if case.get('background'):
                yield case['background']


def _map_paths(session, convert):
    for source in _sources(session):
        if 'path' in source:
            source['path'] = convert(source['path'])
        options = source.get('options', {})
        if options.get('h5_file'):
            options['h5_file'] = convert(options['h5_file'])


def save_session(session, path):
    """Save a validated recipe atomically; source arrays are never embedded.

    Parameters
    ----------
    session : dict
        Committed recipe, optionally containing explicitly labelled draft edits.
    path : str or Path
        Destination JSON file. Input paths are stored relative to this file.
    """
    from .. import __version__

    result = validate_session(session)
    destination = Path(path).resolve()
    _map_paths(result, lambda p: os.path.relpath(Path(p).resolve(), destination.parent))
    result['saved_with'] = dict(mageometry=__version__, python=platform.python_version())
    for package in ('numpy', 'scipy', 'pyvista', 'vtk', 'PySide6', 'pyvistaqt', 'h5py'):
        try:
            result['saved_with'][package] = version(package)
        except PackageNotFoundError:
            pass
    fd, temporary = tempfile.mkstemp(prefix=destination.name + '.', suffix='.tmp', dir=destination.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_session(path):
    """Read a recipe without executing code or loading its magnetic arrays.

    Parameters
    ----------
    path : str or Path
        Saved session JSON file.

    Returns
    -------
    dict
        Validated recipe with absolute source paths. Missing files can be
        relocated in the source editor before preparation.
    """
    source = Path(path).resolve()
    result = json.loads(source.read_text(encoding='utf-8'))
    _map_paths(result, lambda p: str((source.parent / p).resolve()))
    return validate_session(result)
