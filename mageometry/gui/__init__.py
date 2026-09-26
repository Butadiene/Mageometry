"""Optional desktop application. Importing this module does not import Qt."""


def launch(session=None):
    """Launch the desktop workspace with an optional session recipe.

    Parameters
    ----------
    session : dict, optional
        Portable session recipe; defaults to a T96 snapshot.

    Returns
    -------
    int or MainWindow
        Event-loop exit code, or a window when an application already exists.
    """
    from .app import run
    return run(session)
