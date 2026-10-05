"""Launch five T96 + dipole cases at Dst=-30 nT and IMF Bz=-10 nT.

The default IMF By scan is -10, -5, 0, 5, 10 nT.
Use --by to change the scan, or --evaluation grid for grid interpolation.
The importable model helpers use these same comparison conditions.
"""

from mageometry.session import presets as _presets
from mageometry.session.presets import (
    DEFAULT_BY, DST, IMF_BZ, DEFAULT_SHAPE, MODEL_DELTA, EPOCH, INNER_RADIUS, PDYN,
    case_label, inner_mask,
)
from mageometry.gui.app import main as _main


def make_cases(by_values=DEFAULT_BY, shape=DEFAULT_SHAPE):
    """Sample the comparison at Dst=-30 nT and IMF Bz=-10 nT.

    Parameters
    ----------
    by_values : sequence of float, optional
        IMF By inputs in nT; defaults to -10, -5, 0, 5, 10.
    shape : tuple of int, optional
        Grid node counts in x/y/z, each at least three.

    Returns
    -------
    dict of str to GriddedField
        Independent snapshots in GSM Re and nT.
    """
    return _presets.make_cases(by_values, shape, dst=DST, bz=IMF_BZ)


def make_fields(by_values=DEFAULT_BY):
    """Build direct evaluators at Dst=-30 nT and IMF Bz=-10 nT.

    Parameters
    ----------
    by_values : sequence of float, optional
        IMF By inputs in nT; defaults to -10, -5, 0, 5, 10.

    Returns
    -------
    dict of str to callable
        Independent field functions in GSM Re and nT, for serial evaluation.
    """
    return _presets.make_fields(by_values, dst=DST, bz=IMF_BZ)


def main(argv=None):
    return _main(argv, default_by=DEFAULT_BY, description=__doc__)


if __name__ == '__main__':
    main()
