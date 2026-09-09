"""Explore magnetic geometry; shares loading and controls with fac_viewer.

    python examples/geometry_viewer.py --component sigma --slice x --slice-only
    python examples/geometry_viewer.py --xmf run000.xmf --component gamma
"""

from fac_viewer import main


if __name__ == '__main__':
    main(default_component='alpha', description=__doc__)
