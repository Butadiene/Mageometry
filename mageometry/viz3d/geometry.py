"""General magnetic geometry viewer using the shared current/slice renderer."""

from .fac import current_view, _overview_view
from ._current import _component_name
from ._overview_data import _validate_cases, _validate_fields


def geometry_view(gridded_field, component='alpha', *, background_choices=None,
                  background_loader=None, background_directory=None, **kwargs):
    """Explore magnetic currents and transverse field-line structure.

    Parameters
    ----------
    gridded_field : GriddedField
        Magnetic snapshot in consistent Cartesian coordinates.
    component : str, optional
        Initial diagnostic: ``alpha`` (default), ``beta_g``, ``delta_g``, ``gamma``,
        ``omega_c``, ``eta``, or any component accepted by :func:`current_view`.
    background_choices : mapping of str to callable or GriddedField, optional
        Named background presets for the GUI. No background is inferred from
        metadata. The BACKGROUND menu also provides None and a file browser.
    background_loader : callable, optional
        ``loader(path) -> GriddedField`` for GUI file selection. Defaults to
        XDMF or VTK loading at full resolution. Supply a loader with the same
        stride/region as the total input when it has been subsetted.
    background_directory : str or Path, optional
        Initial folder in the in-viewer browser; default current directory.
    **kwargs
        Options forwarded to :func:`current_view`, including ``field``,
        ``geometry_delta``, ``slice_normal``, ``slice_only``, and ``show``.
        All five transverse rates have inverse-length units and ignore
        ``current_scale``. Eta is dimensionless and also ignores scaling.
        These diagnostics do not draw current arrows.

    Returns
    -------
    pyvista.Plotter
        Shared interactive viewer with a diagnostic dropdown, signed peak
        projections, threshold regions, and movable slices. Gamma is
        nonnegative and occupies the positive half of the shared colour scale.
        Eta defaults to a fixed [-1, 1] scale and is NaN where both alpha
        and gamma are zero. Optional source metadata is shown in both modes.
        Background selection enables total/background/residual contributions
        (F7/F8) with shared scales; None restores all current components.
        Enabling a background while viewing a current component selects eta.
        Total magnetic lines, cameras, slices and thresholds are retained.
        An invalid background leaves the displayed data unchanged.

    Notes
    -----
    Transverse rates use ``geometry.field_line_transverse_geometry``.
    Alpha and gamma are defined even where the Frenet frame is undefined;
    beta_g and delta_g need nonzero curvature. At weak curvature use gamma to
    assess anisotropy and check resolution/derivative-step convergence.
    Legacy notebook current components keep their original discretization
    and validity masks, so their alpha estimates can differ numerically.
    """
    return _overview_view(gridded_field, component=_component_name(component), background_controls=True,
                          background_choices=background_choices,
                          background_loader=background_loader,
                          background_directory=background_directory, **kwargs)


def compare_geometry(cases, component='alpha', *, initial_case=None,
                     color_limits=None, cache_size=2, fields=None, delta=None,
                     **kwargs):
    """Compare labelled magnetic snapshots under shared viewing conditions.

    Parameters
    ----------
    cases : mapping of str to GriddedField
        Ordered, nonempty collection on exactly the same x/y/z axes. The
        caller must ensure common coordinates, units and preprocessing.
        Conflicting ``coordinate_system``, ``length_unit`` or ``field_unit``
        metadata are rejected. Input grids must not change while viewing.
    component : str, optional
        Initial diagnostic, as in :func:`geometry_view`.
    initial_case : str, optional
        Initially displayed label; defaults to the first case. The first
        case always supplies default thresholds and automatic line seeds.
    color_limits : mapping of str to float, optional
        Positive symmetric colour limits in displayed units per diagnostic.
        Eta defaults to 1 (the full dimensionless range). Other diagnostics
        use the largest per-case 98th percentile of finite
        absolute values, falling back to the peak or 1 for zero/empty data.
    cache_size : int, optional
        Maximum retained case previews, default 2. Evicted cases are
        recomputed. Source grids and the displayed scene are held separately.
    fields : mapping of str to callable, optional
        Direct ``field(x, y, z) -> (bx, by, bz)`` evaluators with exactly
        the same labels as ``cases``. All cases then use direct evaluation
        for magnetic derivatives and tracing, while grids define display
        coordinates and valid base nodes. Callables must match their grids'
        coordinates, units and magnetic fields and remain reproducible
        after other cases are evaluated. The caller owns model state.
        When omitted, all cases use linear preview-grid interpolation.
    delta : float or (3,) array_like, optional
        Positive finite Cartesian difference step(s), shared by all cases.
        Required with ``fields`` and unsupported without it. Geometry
        diagnostics default to ``min(delta)``; ``geometry_delta`` overrides
        their step without changing the independent Cartesian FAC estimate.
    **kwargs
        Shared :func:`geometry_view` options, including ``threshold``,
        ``geometry_delta``, ``mask``, ``seeds``, ``slice_panel`` and ``show``.
        The single-case ``field`` option is unsupported; use ``fields``
        to supply one callable per case.

    Returns
    -------
    pyvista.Plotter
        Viewer with independent dataset (F7/F8) and diagnostic (F5/F6)
        dropdowns. Dataset changes preserve cameras, slices, colour limits
        and the absolute threshold, and retrace lines from fixed seeds.

    Notes
    -----
    The first use of each diagnostic evaluates all cases synchronously to
    establish shared limits and slider bounds. Threshold defaults use the
    first case's ``percentile``; subsequent adjustments are per diagnostic
    and shared across cases. NaN values remain blank. ``current_scale`` and
    unit labels apply equally to every case and never convert metadata.
    """
    cases = _validate_cases(cases)
    if component is None:
        raise ValueError('component must name a diagnostic.')
    if initial_case is None:
        initial_case = next(iter(cases))
    if initial_case not in cases:
        raise ValueError(f'Unknown case {initial_case!r}.')
    if kwargs.pop('field', None) is not None:
        raise ValueError('field is unsupported for comparison; use per-case fields.')
    fields = _validate_fields(cases, fields, delta)
    return _overview_view(
        next(iter(cases.values())), component=component, delta=delta,
        comparison=dict(cases=cases, initial_case=initial_case,
                        color_limits=color_limits, cache_size=cache_size,
                        fields=fields), **kwargs)


def transverse_contribution_view(gridded_field, background, component='eta', *,
                                 contribution='total', background_label='Background',
                                 color_limits=None, **kwargs):
    """Compare total, background and residual gradients in a common frame.

    Parameters
    ----------
    gridded_field : GriddedField
        Total magnetic field defining the display coordinates and valid nodes.
    background : callable or GriddedField
        Background in identical coordinates and units. Background grids must
        have identical axes and compatible declared units. Both grids use the
        same preview sampling; callables use the total geometry difference step.
    component : str, optional
        One of alpha, beta_g, delta_g, gamma, omega_c or eta (default).
    contribution : {'total', 'background', 'residual'}, optional
        Initial contribution. F7/F8 or the left dropdown changes it.
    background_label : str, optional
        Declared background identity shown in source metadata.
    color_limits : mapping of str to float, optional
        Shared symmetric limits, as in :func:`compare_geometry`.
    **kwargs
        :func:`geometry_view` options, including ``background_choices``,
        ``background_loader`` and ``background_directory`` for GUI selection.
        ``field`` optionally supplies the total
        evaluator; ``geometry_delta`` controls both gradient stencils. Without
        evaluators, derivatives use linear preview-grid interpolation.

    Returns
    -------
    pyvista.Plotter
        Viewer retaining total field lines, cameras, slices and per-diagnostic
        thresholds across contributions. Scales cover all three contributions;
        eta defaults to [-1, 1]. Default thresholds and seeds use the total.

    Notes
    -----
    Uses :func:`mageometry.geometry.field_line_transverse_decomposition`.
    The residual is formed before projection and scalar diagnostics. Its frame
    and normalization still contain the background through the total field.
    Contribution eta and omega_c characterize projected gradient operators,
    not the actual field-line geometry of a standalone residual field.
    """
    from ._contribution_data import CONTRIBUTIONS, _transverse_component

    _transverse_component(component)
    if contribution not in CONTRIBUTIONS:
        raise ValueError(f'Unknown contribution {contribution!r}.')
    return _overview_view(gridded_field, component=component, background_controls=True,
                          contributions=dict(background=background,
                                             background_label=background_label,
                                             initial_case=contribution,
                                             color_limits=color_limits), **kwargs)
