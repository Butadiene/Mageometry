# mageometry/viz3d/_pv.py
"""Lazy pyvista access and small shared helpers."""


def require_pyvista():
    try:
        import pyvista
    except ImportError:
        raise ImportError(
            "mageometry.viz3d requires the optional dependency pyvista. "
            "Install it with: pip install pyvista (or pip install -e .[viz3d])"
        ) from None
    return pyvista


def get_plotter(plotter=None, **kwargs):
    """Return ``plotter`` or a new ``pyvista.Plotter``."""
    pv = require_pyvista()
    if plotter is not None:
        return plotter
    return pv.Plotter(**kwargs)
