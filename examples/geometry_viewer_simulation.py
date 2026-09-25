"""Explore magnetic geometry in a user-supplied simulation snapshot.

    python examples/geometry_viewer_simulation.py --xmf snapshot.xmf
    python examples/geometry_viewer_simulation.py --xmf snapshot.xmf --stride 4
    python examples/geometry_viewer_simulation.py --vtk snapshot.vti --component beta_g
    python examples/geometry_viewer_simulation.py --h5 field.h5 --origin 0 0 0 --spacing 1 1 1

Specify a snapshot with --xmf, --vtk, or --h5. Relative paths are resolved
from the working directory. No snapshot path or grid size is built in.
All nodes are read by default; use --stride to reduce memory use.
Coordinates and geometry rates retain the input grid's units.
All display controls are shared with geometry_viewer.
"""

from geometry_viewer import main


if __name__ == '__main__':
    main(default_component='alpha', description=__doc__,
         require_source=True)
