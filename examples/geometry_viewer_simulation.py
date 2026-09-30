"""Launch the desktop workspace with an explicit snapshot (requires gui).

Supply --xmf, --vtk or --h5 (with --origin and --spacing for direct HDF5).
The common ``python -m mageometry.gui`` CLI accepts the same file options.
There is no built-in simulation path, grid size or unit conversion.
"""

from mageometry.gui.app import main as _main


def main(argv=None):
    return _main(argv, require_source=True, description=__doc__)


if __name__ == '__main__':
    main()
