"""General magnetic geometry viewer using the shared current/slice renderer."""

from .fac import current_view


def geometry_view(gridded_field, component='alpha', **kwargs):
    """Explore magnetic currents and transverse field-line structure.

    Parameters
    ----------
    gridded_field : GriddedField
        Magnetic snapshot in consistent Cartesian coordinates.
    component : str, optional
        Initial diagnostic: ``alpha`` (default), ``sigma``, ``q``, ``gamma``,
        ``omega_c``, or any component accepted by :func:`current_view`.
    **kwargs
        Options forwarded to :func:`current_view`, including ``field``,
        ``geometry_delta``, ``slice_normal``, ``slice_only``, and ``show``.
        All five transverse rates have inverse-length units and ignore
        ``current_scale``. They do not draw current arrows.

    Returns
    -------
    pyvista.Plotter
        Shared interactive viewer with a diagnostic dropdown, signed peak
        projections, threshold regions, and movable slices. Gamma is
        nonnegative and occupies the positive half of the shared colour scale.

    Notes
    -----
    Transverse rates use ``geometry.field_line_transverse_geometry``.
    Alpha and gamma are defined even where the Frenet frame is undefined;
    sigma and q need nonzero curvature. At weak curvature use gamma to
    assess anisotropy and check resolution/derivative-step convergence.
    Legacy notebook current components keep their original discretization
    and validity masks, so their alpha estimates can differ numerically.
    """
    return current_view(gridded_field, component=component, **kwargs)
