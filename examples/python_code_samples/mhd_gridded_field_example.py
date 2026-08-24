"""
Example: field line geometry analysis of MHD simulation output.

Loads a gridded magnetic field from an XDMF-described HDF5 file (as written
by many magnetosphere MHD codes) and runs the Mageometry geometry functions
on the interpolated field. All positions and geometry results are in the
simulation's own grid units.

Usage:
    python mhd_gridded_field_example.py --xmf run000.xmf [--h5 run000-heavy.h5]
    python mhd_gridded_field_example.py --h5 data.h5 --origin 0 0 0 --spacing 1 1 1

The --h5 option overrides the HDF5 path referenced inside the XDMF file
(useful when the heavy-data file was renamed or moved). Without --xmf, the
HDF5 file is read directly and --origin/--spacing supply the grid geometry.
"""

import argparse

import numpy as np

from mageometry import (
    load_xdmf,
    load_hdf5,
    field_line_curvature,
    field_line_frenet_frame,
    verify_unit_vectors,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xmf', help='XDMF metadata file describing the grid')
    p.add_argument('--h5', help='HDF5 heavy-data file (override or direct read)')
    p.add_argument('--origin', nargs=3, type=float, default=[0.0, 0.0, 0.0],
                   help='grid origin (x y z), direct HDF5 read only')
    p.add_argument('--spacing', nargs=3, type=float, default=[1.0, 1.0, 1.0],
                   help='grid spacing (dx dy dz), direct HDF5 read only')
    p.add_argument('--center', nargs=3, type=float, default=None,
                   help='planet/dipole center in grid coordinates '
                        '(default: |B| maximum on a coarse subsample)')
    args = p.parse_args()
    if not args.xmf and not args.h5:
        p.error('give --xmf (optionally with --h5) or --h5 alone')
    return args


def find_center(grid, stride=4):
    """Estimate the dipole center as the |B| maximum (coarse scan + local refine)."""
    b = grid.b[::stride, ::stride, ::stride, :]
    bmag = np.sqrt((b.astype(np.float64) ** 2).sum(axis=-1))
    coarse = stride * np.array(np.unravel_index(np.argmax(bmag), bmag.shape))
    sl = tuple(slice(max(0, c - 2 * stride), c + 2 * stride) for c in coarse)
    local = grid.b[sl + (slice(None),)].astype(np.float64)
    lmag = np.sqrt((local ** 2).sum(axis=-1))
    i, j, k = (s.start + li for s, li in
               zip(sl, np.unravel_index(np.argmax(lmag), lmag.shape)))
    return grid.x[i], grid.y[j], grid.z[k]


def main():
    args = parse_args()

    if args.xmf:
        grid = load_xdmf(args.xmf, h5_file=args.h5)
    else:
        grid = load_hdf5(args.h5, origin=tuple(args.origin),
                         spacing=tuple(args.spacing))
    print(f"Loaded: {grid}")

    cx, cy, cz = args.center if args.center else find_center(grid)
    print(f"Dipole center (grid coords): ({cx:.1f}, {cy:.1f}, {cz:.1f})")

    field = grid.field(method='linear')
    dx = grid.x[1] - grid.x[0]  # use the grid step to scale FD deltas

    # --- Field values around the center ---
    print("\nField samples (grid units):")
    for off in (5, 10, 20):
        bx, by, bz = field(cx, cy + off * dx, cz)
        bmag = np.hypot(np.hypot(bx, by), bz)
        print(f"  y-offset {off:3d} cells: |B| = {bmag:10.4g}   "
              f"B = ({bx:9.3g}, {by:9.3g}, {bz:9.3g})")

    # --- Curvature profile vs. dipole expectation 3/r ---
    print("\nCurvature along +y from the center (kappa*r/3 -> 1 for a dipole):")
    print("   r [cells]   kappa [1/unit]    3/r     kappa*r/3")
    for cells in (8, 12, 16, 24, 32, 48):
        r = cells * dx
        kappa = field_line_curvature(field, cx, cy + r, cz, delta=dx)
        print(f"   {cells:6d}     {kappa:12.5f} {3/r:9.5f}   {kappa*r/3:8.3f}")

    # --- Frenet frame quality on a ring around the center ---
    phi = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    r = 16 * dx
    x = np.full_like(phi, cx)
    y = cy + r * np.cos(phi)
    z = cz + r * np.sin(phi)
    frame = field_line_frenet_frame(field, x, y, z, delta=dx)
    errors = verify_unit_vectors(*frame[:9])
    kappa = frame[9]
    # The normal is NaN where it cannot be defined reliably (straight or
    # numerically degenerate field lines) — exclude those points.
    valid = np.isfinite(frame[3])
    max_err = max(np.max(np.abs(e[valid])) for e in errors.values())
    print(f"\nFrenet frame on r={r:.1f} ring: {valid.sum()}/{phi.size} points valid, "
          f"max orthonormality error {max_err:.2e}")
    print(f"curvature on ring: min {kappa[valid].min():.4f}, "
          f"max {kappa[valid].max():.4f} [1/grid-unit]")


if __name__ == '__main__':
    main()
