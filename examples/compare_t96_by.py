"""Launch the desktop workspace with six IMF By cases.

Equivalent to ``python -m mageometry.gui --by -5 -3 -1 1 3 5``.
Use --by to change the scan, or --evaluation grid for grid interpolation.
The model helpers remain importable here for existing notebooks.
"""

from mageometry.session.presets import (
    DEFAULT_BY, DEFAULT_SHAPE, MODEL_DELTA, EPOCH, INNER_RADIUS, PDYN, DST,
    IMF_BZ, case_label, inner_mask, make_cases, make_fields,
)
from mageometry.gui.app import main as _main


def main(argv=None):
    return _main(argv, default_by=DEFAULT_BY, description=__doc__)


if __name__ == '__main__':
    main()
