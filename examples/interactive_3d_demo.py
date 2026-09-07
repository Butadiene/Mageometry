"""
Interactive 3D visualization demo (mageometry.viz3d, requires pyvista).

Samples a T96 + dipole magnetosphere onto a grid and opens two interactive
desktop windows in sequence (close one to get the next):

1. `fac_view` — signed FAC regions, current direction arrows, magnetic
   field lines and three peak-projection maps. Drag the threshold slider
   to isolate the stronger currents. Red flows along B, blue against B.
2. `slice_view` in free-plane mode — curvature on a single plane widget
   (drag the arrow to translate, grab the plane to rotate), with Frenet
   frame arrows along the noon meridian. The right panel views the slice
   face-on, its camera tracking the plane normal as you rotate it.

Camera: left-drag rotates, wheel zooms, shift+drag pans. Run with:

    pip install -e .[viz3d]
    python examples/interactive_3d_demo.py

On WSL2: if the window title/renderer reports "llvmpipe" (software
rendering), set ``GALLIUM_DRIVER=d3d12`` to render on the Windows GPU:

    GALLIUM_DRIVER=d3d12 python examples/interactive_3d_demo.py
"""

import numpy as np

from mageometry import GriddedField, geopack, geopack_field, viz3d

# --- Build the field and sample it onto a grid ------------------------------
ps = geopack.recalc(100)                                # epoch -> dipole tilt
parmod = [2.0, -20.0, 0.0, -5.0, 0, 0, 0, 0, 0, 0]      # Pdyn, Dst, ByIMF, BzIMF
field = geopack_field('t96', 'dip', parmod, ps)

x = np.linspace(-15.0, 5.0, 81)
y = np.linspace(-8.0, 8.0, 65)
z = np.linspace(-6.0, 6.0, 49)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
with np.errstate(divide='ignore', invalid='ignore'):    # dipole singular at r=0
    bx, by, bz = field(X, Y, Z)
# Blank the near-Earth region: the model field diverges there and would
# dominate every colour scale. NaN stays blank in all viz3d views.
inside = X ** 2 + Y ** 2 + Z ** 2 < 2.0 ** 2
for comp in (bx, by, bz):
    comp[inside] = np.nan
grid = GriddedField(x, y, z, bx, by, bz)

# --- 1. FAC overview -------------------------------------------------------
print("Window 1: FAC regions and current directions "
      "(close the window to continue)")
viz3d.fac_view(grid, field=field, delta=0.002,
                mask=lambda x, y, z: x*x + y*y + z*z < 2.0**2,
                trace_kwargs={'ds': 0.1, 'r0': 2.0}, planet_radius=1.0,
                length_unit='Re', current_scale=0.125,
                current_label='J parallel [nA/m^2]')

# --- 2. Free slice plane + Frenet frames ------------------------------------
print("Window 2: curvature on a free slice plane + Frenet frames")
plotter = viz3d.slice_view(grid, 'curvature', mode='plane', normal='y',
                           field=field, delta=0.1, show=False)
viz3d.add_frenet_frame(plotter, field, [-4.0, -6.0, -8.0], [0.0, 0.0, 0.0],
                       [0.5, 1.0, 1.5], delta=0.1, length=1.2)
plotter.show()
