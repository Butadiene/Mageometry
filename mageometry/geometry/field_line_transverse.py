"""Local transverse rotation and shear from first magnetic-field derivatives."""

import numpy as np

from .field_line_geometry import _as_arrays, _finish


def field_line_transverse_geometry(field, x, y, z, delta=0.01, curvature_tol=0.0):
    """Decompose the transverse gradient of the unit magnetic direction.

    Parameters
    ----------
    field : callable
        ``field(x, y, z) -> (bx, by, bz)`` accepting broadcast NumPy arrays.
    x, y, z : float or array_like
        Coordinates in the field's length unit; inputs are broadcast.
    delta : float or (3,) array_like, optional
        Positive finite Cartesian central-difference steps. Default 0.01.
    curvature_tol : float, optional
        Nonnegative curvature cutoff (inverse length). Beta_g and delta_g are NaN
        at or below this cutoff. Default zero; increase for noisy data.

    Returns
    -------
    dict of float or ndarray
        ``alpha = p-q``, ``beta_g = p+q``, ``delta_g = a-d``,
        ``gamma = sqrt(beta_g**2+delta_g**2)``, and ``omega_c`` in inverse
        length, where p=b.dot(partial_n T), q=n.dot(partial_b T),
        a=n.dot(partial_n T), d=b.dot(partial_b T), and b=T cross n.
        ``omega_c = sign(alpha)*sqrt(max(alpha**2-gamma**2, 0))/2``.
        ``eta = (alpha**2-gamma**2)/(alpha**2+gamma**2)`` is dimensionless,
        in [-1, 1], and NaN where alpha and gamma are both zero.
        ``curvature`` is also returned in inverse length. Scalar coordinates
        produce scalar values. Nulls and invalid derivative stencils give NaN.
        The ``delta`` argument is a numerical step,
        unrelated to the returned physical diagnostic ``delta_g``.

    Notes
    -----
    Only first derivatives of B are sampled; no derivatives of n are used.
    With P=I-T T^t and G=grad(B), L=P G P/|B| is the Cartesian transverse
    map; M is its 2D representation in the (n, b) basis.
    Gamma is computed from its symmetric traceless part in Cartesian space,
    so alpha, gamma, omega_c and eta do not require curved field lines.
    Beta_g and delta_g refer to the local curvature normal and become unstable
    at weak curvature. The rates describe geometry per length, not current
    densities or finite-distance winding numbers. Notation follows the
    project's baseline in ``docs/fac_anisotropy_theory.md``. The coiling
    convention follows Tassev & Savcheva (2019), https://arxiv.org/abs/1901.00865.
    """
    scalar, reference, gradient, valid = _reference(field, x, y, z, delta, curvature_tol)
    result = _contribution(gradient, reference, valid)
    result = {key: result[key] for key in
              ('alpha', 'beta_g', 'delta_g', 'gamma', 'omega_c', 'eta')}
    result['curvature'] = reference['curvature']
    return _finish_mapping(scalar, result)


def _sample(field, coords, steps):
    def evaluate(points):
        components = tuple(field(*points))
        if len(components) != 3:
            raise ValueError('field must return three magnetic components.')
        return np.stack(np.broadcast_arrays(
            *[np.asarray(c, dtype=float) for c in components], coords[0])[:3], axis=-1)

    magnetic = evaluate(coords)
    derivatives = []
    for axis, step in enumerate(steps):
        plus, minus = list(coords), list(coords)
        plus[axis] = coords[axis] + step
        minus[axis] = coords[axis] - step
        derivatives.append((evaluate(plus) - evaluate(minus)) / (2 * step))
    return magnetic, np.stack(derivatives, axis=-1)


def _reference(field, x, y, z, delta, curvature_tol):
    steps = np.broadcast_to(np.asarray(delta, dtype=float), (3,))
    if not np.all(np.isfinite(steps) & (steps > 0)):
        raise ValueError('delta must contain positive finite step sizes.')
    if not np.isscalar(curvature_tol) or not np.isfinite(curvature_tol) or curvature_tol < 0:
        raise ValueError('curvature_tol must be a nonnegative finite scalar.')
    scalar, x, y, z = _as_arrays(x, y, z)
    magnetic, gradient = _sample(field, (x, y, z), steps)
    magnitude = np.linalg.norm(magnetic, axis=-1)
    valid = (magnitude > 0) & np.isfinite(magnitude) & np.all(np.isfinite(gradient), axis=(-2, -1))
    with np.errstate(divide='ignore', invalid='ignore'):
        tangent = magnetic / magnitude[..., None]
        projection = np.eye(3) - tangent[..., :, None] * tangent[..., None, :]
        direction_gradient = (projection @ gradient) / magnitude[..., None, None]
        curvature_vector = (direction_gradient @ tangent[..., None])[..., 0]
        curvature = np.linalg.norm(curvature_vector, axis=-1)
        normal = curvature_vector / np.where(curvature > curvature_tol, curvature, np.nan)[..., None]
        binormal = np.cross(tangent, normal)
    reference = dict(magnetic_field=magnetic, field_magnitude=magnitude,
                     tangent=tangent, normal=normal, binormal=binormal,
                     projection=projection, curvature=curvature)
    return scalar, _mask_mapping(reference, valid), gradient, valid


def _mask_mapping(values, valid):
    return {key: np.where(valid.reshape(valid.shape + (1,) * (value.ndim - valid.ndim)),
                          value, np.nan) for key, value in values.items()}


def _finish_mapping(scalar, values):
    # _as_arrays adds a length-one point axis for scalar inputs.
    return {key: (value[0] if value.ndim > 1 else _finish(True, value))
            if scalar else value for key, value in values.items()}


def _contribution(gradient, reference, valid):
    tangent, normal, binormal = (reference[k] for k in ('tangent', 'normal', 'binormal'))
    projection, magnitude = reference['projection'], reference['field_magnitude']
    with np.errstate(divide='ignore', invalid='ignore'):
        direction_gradient = (projection @ gradient) / magnitude[..., None, None]
        transverse = direction_gradient @ projection
        symmetric = (transverse + np.swapaxes(transverse, -1, -2)) / 2
        dilation = np.trace(transverse, axis1=-2, axis2=-1)
        shear = symmetric - dilation[..., None, None] * projection / 2
        gamma = np.sqrt(2 * np.sum(shear * shear, axis=(-2, -1)))
        curl = np.stack((gradient[..., 2, 1] - gradient[..., 1, 2],
                         gradient[..., 0, 2] - gradient[..., 2, 0],
                         gradient[..., 1, 0] - gradient[..., 0, 1]), axis=-1)
        alpha = np.sum(tangent * curl, axis=-1) / magnitude
        dn = (direction_gradient @ normal[..., None])[..., 0]
        db = (direction_gradient @ binormal[..., None])[..., 0]
        beta_g = np.sum(binormal * dn + normal * db, axis=-1)
        delta_g = np.sum(normal * dn - binormal * db, axis=-1)
        omega_c = np.sign(alpha) * np.sqrt(np.maximum(alpha**2 - gamma**2, 0)) / 2
        # Normalize before squaring; the 0/0 case remains undefined.
        scale = np.maximum(np.abs(alpha), gamma)
        scaled_alpha, scaled_gamma = alpha / scale, gamma / scale
        eta = ((scaled_alpha**2 - scaled_gamma**2)
               / (scaled_alpha**2 + scaled_gamma**2))
    return _mask_mapping(dict(gradient=gradient, transverse=transverse, shear=shear,
                              trace=dilation, divergence=np.trace(gradient, axis1=-2, axis2=-1),
                              alpha=alpha, beta_g=beta_g, delta_g=delta_g, gamma=gamma,
                              omega_c=omega_c, eta=eta), valid)


def field_line_transverse_decomposition(field, background, x, y, z, delta=0.01,
                                        curvature_tol=0.0):
    """Resolve gradient contributions in the total magnetic field's frame.

    Parameters
    ----------
    field, background : callable
        Total and background ``field(x, y, z) -> (bx, by, bz)`` evaluators.
        Both must accept broadcast arrays in identical coordinates and units.
        A dipole, IGRF, or a simulation reference field may be the background.
    x, y, z : float or array_like
        Coordinates, broadcast together.
    delta : float or (3,) array_like, optional
        Positive finite central-difference steps, shared by both fields.
    curvature_tol : float, optional
        Nonnegative cutoff for the total-field curvature normal.

    Returns
    -------
    dict
        ``reference`` contains total ``magnetic_field``, ``field_magnitude``,
        ``tangent``, ``normal``, ``binormal``, ``projection`` and ``curvature``.
        ``total``, ``background`` and ``residual`` each contain ``gradient``
        (G_ij = partial_j B_i), ``transverse`` (P G P / |B_total|), ``shear``
        (symmetric transverse traceless part), ``trace`` of the transverse
        map, ``divergence`` of B, and ``alpha``, ``beta_g``, ``delta_g``,
        ``gamma``, ``omega_c``, ``eta`` with the conventions of
        :func:`field_line_transverse_geometry`. Vector and tensor dimensions
        follow the broadcast point dimensions. Scalars return floats and
        vectors/tensors of shape (3,)/(3, 3).

    Notes
    -----
    The residual gradient is G_total - G_background. All contributions use
    the TOTAL field's magnitude, projector and Frenet frame. Gradient,
    transverse, shear, trace, divergence, alpha, beta_g and delta_g are
    additive where defined; gamma, eta and omega_c are not. Contribution
    eta/omega_c describe projected operators, not actual winding of residual
    or background field lines. To study a residual field's own geometry,
    pass that field separately to ``field_line_transverse_geometry``.

    Background nulls are allowed: no division by |B_background| occurs.
    Invalid background samples/stencils invalidate background and residual
    only. Total nulls or invalid total stencils invalidate all outputs.
    Beta_g/delta_g additionally require resolved total curvature. A uniform
    external field has zero residual gradient but can change the reference
    frame and total geometry; this is attribution, not a dipole-independent
    counterfactual. Divergence has magnetic-field/length units; other scalar
    rates and transverse/shear tensors have inverse-length units.
    """
    scalar, reference, gradient, valid = _reference(field, x, y, z, delta, curvature_tol)
    _, x, y, z = _as_arrays(x, y, z)
    steps = np.broadcast_to(np.asarray(delta, dtype=float), (3,))
    magnetic_bg, gradient_bg = _sample(background, (x, y, z), steps)
    valid_bg = (valid & np.all(np.isfinite(magnetic_bg), axis=-1)
                & np.all(np.isfinite(gradient_bg), axis=(-2, -1)))
    result = dict(reference=reference,
                  total=_contribution(gradient, reference, valid),
                  background=_contribution(gradient_bg, reference, valid_bg),
                  residual=_contribution(gradient - gradient_bg, reference, valid_bg))
    return {key: _finish_mapping(scalar, values) for key, values in result.items()}
