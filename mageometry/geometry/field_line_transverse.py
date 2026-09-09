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
        Nonnegative curvature cutoff (inverse length). Sigma and q are NaN
        at or below this cutoff. Default zero; increase for noisy data.

    Returns
    -------
    dict of float or ndarray
        ``alpha = a-c``, ``sigma = a+c``, ``q = u-v``,
        ``gamma = sqrt(sigma**2+q**2)``, and ``omega_c`` in inverse length,
        where a=b.dot(partial_n T), c=n.dot(partial_b T),
        u=n.dot(partial_n T), v=b.dot(partial_b T), and b=T cross n.
        ``omega_c = sign(alpha)*sqrt(max(alpha**2-gamma**2, 0))/2``.
        ``curvature`` is also returned in inverse length. Scalar coordinates
        produce scalar values. Nulls and invalid derivative stencils give NaN.

    Notes
    -----
    Only first derivatives of B are sampled; no derivatives of n are used.
    With P=I-T T^t and G=grad(B), M=P G P/|B| is the transverse map.
    Gamma is computed from its symmetric traceless part in Cartesian space,
    so alpha, gamma and omega_c remain defined on straight field lines.
    Sigma and q refer to the local curvature normal and become unstable at
    weak curvature. These are rates per length, not current densities or
    finite-distance winding numbers. The coiling convention follows
    Tassev & Savcheva (2019), https://arxiv.org/abs/1901.00865.
    """
    steps = np.broadcast_to(np.asarray(delta, dtype=float), (3,))
    if not np.all(np.isfinite(steps) & (steps > 0)):
        raise ValueError('delta must contain positive finite step sizes.')
    if not np.isscalar(curvature_tol) or not np.isfinite(curvature_tol) or curvature_tol < 0:
        raise ValueError('curvature_tol must be a nonnegative finite scalar.')
    scalar, x, y, z = _as_arrays(x, y, z)
    coords = (x, y, z)

    def evaluate(points):
        return np.stack(np.broadcast_arrays(
            *[np.asarray(c, dtype=float) for c in field(*points)], points[0])[:3], axis=-1)

    magnetic = evaluate(coords)
    derivatives = []
    for axis, step in enumerate(steps):
        plus, minus = list(coords), list(coords)
        plus[axis] = coords[axis] + step
        minus[axis] = coords[axis] - step
        derivatives.append((evaluate(plus) - evaluate(minus)) / (2 * step))
    gradient = np.stack(derivatives, axis=-1)  # component, derivative direction
    magnitude = np.linalg.norm(magnetic, axis=-1)
    valid = (magnitude > 0) & np.isfinite(magnitude) & np.all(np.isfinite(gradient), axis=(-2, -1))
    with np.errstate(divide='ignore', invalid='ignore'):
        tangent = magnetic / magnitude[..., None]
        projection = np.eye(3) - tangent[..., :, None] * tangent[..., None, :]
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
        curvature_vector = (direction_gradient @ tangent[..., None])[..., 0]
        curvature = np.linalg.norm(curvature_vector, axis=-1)
        normal = curvature_vector / np.where(curvature > curvature_tol, curvature, np.nan)[..., None]
        binormal = np.cross(tangent, normal)
        dn = (direction_gradient @ normal[..., None])[..., 0]
        db = (direction_gradient @ binormal[..., None])[..., 0]
        sigma = np.sum(binormal * dn + normal * db, axis=-1)
        q = np.sum(normal * dn - binormal * db, axis=-1)
        omega_c = np.sign(alpha) * np.sqrt(np.maximum(alpha**2 - gamma**2, 0)) / 2
    result = dict(alpha=alpha, sigma=sigma, q=q, gamma=gamma,
                  omega_c=omega_c, curvature=curvature)
    return {key: _finish(scalar, np.where(valid, value, np.nan)) for key, value in result.items()}
