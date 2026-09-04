# mageometry/io/vtk.py
"""
VTK file reader (``.vti``, ``.vtr``, and other formats pyvista can read into
an ImageData or RectilinearGrid).

Implements Recipe E of ``docs/simulation_data_formats.md``: VTK stores point
data in x-fastest order, `GriddedField` wants ``(nx, ny, nz)`` arrays with x
slowest, so the components are reshaped and transposed. Requires the
optional ``pyvista`` dependency (imported lazily).
"""

import numpy as np

from .gridded_field import GriddedField, region_slices

__all__ = ["load_vtk"]


def load_vtk(path, name="B", region=None, stride=1):
    """
    Read a VTK file into a `GriddedField`.

    Supports ``ImageData`` (uniform grids, e.g. ``.vti``) and
    ``RectilinearGrid`` (explicit axes, e.g. ``.vtr``). The field can be one
    3-component array or three scalar arrays, as point data (values on the
    grid nodes) or cell data (values placed on the cell centers).

    Parameters
    ----------
    path : str or Path
        The VTK file.
    name : str or (str, str, str), optional
        Name of the 3-component field array (default ``"B"``), or the names
        of the three component arrays, e.g. ``("bx", "by", "bz")``.
    region : sequence, optional
        ``((xmin, xmax), (ymin, ymax), (zmin, zmax))`` sub-box to keep;
        ``None`` per axis keeps the full extent. Unlike the HDF5 readers this
        cannot reduce the read itself — the whole file is read, then sliced
        in memory.
    stride : int or tuple of int, optional
        Keep every ``stride``-th node per axis. Default 1.

    Returns
    -------
    GriddedField

    Notes
    -----
    Curvilinear (``StructuredGrid``) and unstructured meshes carry no
    rectilinear axes and are rejected; see the "Curvilinear, AMR,
    unstructured" section of ``docs/simulation_data_formats.md``.
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError(
            "load_vtk requires the optional dependency pyvista. "
            "Install it with: pip install pyvista (or pip install -e .[viz3d])"
        ) from None

    mesh = pv.read(path)
    if isinstance(mesh, pv.ImageData):
        axes = [o + s * np.arange(n)
                for o, s, n in zip(mesh.origin, mesh.spacing, mesh.dimensions)]
    elif isinstance(mesh, pv.RectilinearGrid):
        axes = [np.asarray(a, dtype=np.float64)
                for a in (mesh.x, mesh.y, mesh.z)]
    else:
        raise ValueError(
            f"{type(mesh).__name__} is not a rectilinear mesh; load_vtk "
            "supports ImageData and RectilinearGrid only."
        )
    nx, ny, nz = mesh.dimensions

    names = (name,) if isinstance(name, str) else tuple(name)
    if len(names) not in (1, 3):
        raise ValueError("name must be one array name or three component names.")
    in_point = all(n in mesh.point_data for n in names)
    in_cell = all(n in mesh.cell_data for n in names)
    if not (in_point or in_cell):
        available = sorted(set(mesh.point_data) | set(mesh.cell_data))
        raise KeyError(f"Array(s) {names} not found; file has {available}.")
    data = mesh.point_data if in_point else mesh.cell_data

    if len(names) == 1:
        arr = np.asarray(data[names[0]])
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"Array {names[0]!r} has shape {arr.shape}; expected "
                "(n, 3). For three scalar arrays pass name=('bx', 'by', 'bz')."
            )
        flat = [arr[:, k] for k in range(3)]
    else:
        flat = [np.asarray(data[n]).ravel() for n in names]

    if in_cell:
        # Cell data sits at the cell centers.
        axes = [0.5 * (a[1:] + a[:-1]) for a in axes]
        nx, ny, nz = nx - 1, ny - 1, nz - 1
    # VTK order is x-fastest -> (nx, ny, nz) with x slowest.
    comps = [c.reshape((nz, ny, nx)).transpose(2, 1, 0) for c in flat]

    sx, sy, sz = region_slices(axes, region, stride)
    return GriddedField(axes[0][sx], axes[1][sy], axes[2][sz],
                        *(c[sx, sy, sz] for c in comps),
                        metadata={"source": str(path), "name": name,
                                  "association": "point" if in_point else "cell"})
