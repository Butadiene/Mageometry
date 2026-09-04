# Simulation Data: Formats and Bringing Your Own (`mageometry.io`)

This document is written for one purpose: to let you get **your** simulation
output — whatever code produced it, whatever it looks like on disk — into
Mageometry, correctly, and to know that you did it correctly.

Part I describes the one data structure everything lands on. Part II is a
cookbook for the formats simulation codes actually produce. Part III covers
the bundled XDMF/HDF5 readers. Part IV is the validation checklist you
should run on any newly written reader, and Part V has practical guidance
on interpolation, tracing, and memory.

---

## Part I — The one contract: `GriddedField`

```python
from mageometry import GriddedField

grid = GriddedField(x, y, z, bx, by, bz, metadata={"source": "run42/step0100"})
field = grid.field(method="linear")     # field(x, y, z) -> (bx, by, bz)
```

That is the whole interface between your data and the analysis library.
Every geometry function (`field_line_curvature`, `field_line_frenet_frame`,
`field_line_directional_derivatives`, ...) and the tracer
(`trace_field_lines`) take the callable that `grid.field()` returns.

| Argument | Requirement |
|---|---|
| `x`, `y`, `z` | 1D arrays, **strictly increasing**, at least 2 points each. Uniform spacing is *not* required — any rectilinear (stretched) axes work. |
| `bx`, `by`, `bz` | Arrays of shape **`(len(x), len(y), len(z))`** — first index is x, last is z. |
| `metadata` | Optional dict, kept as-is (provenance, units, run name, ...). |

What `GriddedField` does **not** do — and therefore what you must do before
constructing it:

- **Units.** Mageometry never interprets units. Positions are in whatever
  unit the axes are in; field values in whatever unit the arrays are in;
  curvature, torsion and directional derivatives come back in 1/(axis unit).
  If you want Re and nT, rescale first (see [Units and coordinates](#units-and-coordinates)).
- **Coordinate system.** The grid is assumed right-handed Cartesian. Which
  way x points (sunward or anti-sunward, GSM or SM or code-native) is your
  business; the geometry is coordinate-independent as long as the three
  axes and the three components refer to the *same* frame.
- **Centering.** Values are taken to sit exactly at the coordinates you
  give. Cell-centered data is fine — just pass cell-center coordinates.
  Staggered (Yee) components must be brought to a common set of points
  first ([Staggered grids](#cell-centered-and-staggered-grids)).
- **Memory.** The three components are stacked into one
  `(nx, ny, nz, 3)` array, so the input arrays are copied once. float32
  input stays float32.

Building blocks provided for your reader (all in `mageometry.io`):

| Tool | Purpose |
|---|---|
| `read_fortran_records(path, dtype)` / `iter_fortran_records` | Fortran unformatted sequential files (record-length markers) |
| `region_slices(axes, region, stride)` | Turn a bounding box / stride into index slices |
| `GriddedField.subvolume(region, stride)` | Cut a sub-box out of an in-memory grid |
| `GriddedField.divergence()` | Sanity check: catches transposed axes, permuted or flipped components |
| `FieldSeries.from_files(paths, loader, times)` | Lazy time series from one file per step and your loader |

---

## Part II — Cookbook: from your files to `GriddedField`

Each recipe is a complete `load_<format>()` function you can copy. They all
end the same way; only the parsing differs. Recipes use only NumPy unless a
third-party reader is the natural tool.

### The mental model: three questions

Before writing any code, answer these for your data; every recipe below is
an instance of them.

1. **What is the memory layout of one component?** Is the fastest-varying
   index x or z (C order `(nz, ny, nx)` vs Fortran order `(nx, ny, nz)`)?
   Are there ghost/guard cells to strip? Any header bytes?
2. **Where are the values located?** Node positions (`origin + i·dx`) or
   cell centers (`origin + (i + ½)·dx`)? Uniform or stretched axes? Are the
   three components at the same locations?
3. **What are the units and the frame?** Code units → physical units;
   which axis is which; is the frame right-handed?

If you get question 1 wrong, `GriddedField.divergence()` will tell you
([Part IV](#part-iv--validating-a-new-reader)).

### Recipe A — NumPy arrays already in memory / `.npy` / `.npz`

```python
import numpy as np
from mageometry import GriddedField

def load_npz(path):
    d = np.load(path)
    # Adjust the key names and the axis order to your file. Here the
    # arrays are stored (nz, ny, nx) as most C/Python codes do.
    bx, by, bz = (d[k].transpose(2, 1, 0) for k in ("bx", "by", "bz"))
    return GriddedField(d["x"], d["y"], d["z"], bx, by, bz,
                        metadata={"source": path})
```

`transpose(2, 1, 0)` is a view — no copy until `GriddedField` stacks the
components.

### Recipe B — Raw binary (`np.fromfile` / `np.memmap`)

Raw dumps are the most common output of home-grown codes. You need to know
(or find out) the element type, byte order, header size, and layout.

```python
import numpy as np
from mageometry import GriddedField

def load_raw(path, nx, ny, nz, dtype=">f4", header_bytes=0,
             variables=("rho", "vx", "vy", "vz", "p", "bx", "by", "bz"),
             origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0)):
    """One file holding `variables` back to back, each stored (nz, ny, nx)."""
    n = nx * ny * nz
    # memmap: nothing is read until it is indexed, so the file may be far
    # larger than memory. dtype includes the byte order ('>' big-endian).
    data = np.memmap(path, dtype=dtype, mode="r", offset=header_bytes,
                     shape=(len(variables), nz, ny, nx))
    comps = []
    for name in ("bx", "by", "bz"):
        k = variables.index(name)
        # .transpose(2, 1, 0) -> (nx, ny, nz); np.ascontiguousarray reads it.
        comps.append(np.ascontiguousarray(data[k].transpose(2, 1, 0)))
    axes = [o + s * np.arange(m) for o, s, m in zip(origin, spacing, (nx, ny, nz))]
    return GriddedField(*axes, *comps, metadata={"source": path})
```

How to find the unknowns when there is no documentation:

- **Byte order / dtype:** read the first few values with both `'<f4'` and
  `'>f4'` (and `f8`); the correct choice gives plausible magnitudes, the
  wrong one gives values like 1e29 or 1e-39.
- **Layout:** file size must equal
  `header + n_variables × nx × ny × nz × itemsize`. If it does not, there
  are ghost cells or per-record markers ([Recipe C](#recipe-c--fortran-unformatted-sequential-files)).
- **Axis order:** try both and run the divergence check.
- **Fortran-order arrays** (x fastest): use `order="F"` in `reshape`, or
  equivalently read as `(nz, ny, nx)` C-order and transpose — the two are
  identical in memory.

### Recipe C — Fortran unformatted sequential files

Fortran `write(unit) array` produces records framed by 4-byte (sometimes
8-byte) length markers. The framing means you cannot `np.fromfile` the
whole thing; `read_fortran_records` handles it and validates the markers
(a mismatch immediately tells you the byte order or marker size is wrong).

```python
import numpy as np
from mageometry import GriddedField
from mageometry.io import read_fortran_records, iter_fortran_records

def load_fortran_planes(path, nx, ny, nz, dtype=">f4",
                        variables=("rho", "vx", "vy", "vz", "p", "bx", "by", "bz"),
                        ny_ghost=(0, 0), origin=(0.0, 0.0, 0.0),
                        spacing=(1.0, 1.0, 1.0)):
    """
    Layout handled here: one record per x-y plane, `nz` planes per variable,
    variables one after another (record k = variable k // nz, plane k % nz).
    Each plane is stored y-major: reshape to (ny + ghosts, nx).
    """
    ny_file = ny + sum(ny_ghost)
    lo, hi = ny_ghost[0], ny_ghost[0] + ny
    comps = {}
    recs = iter_fortran_records(path, dtype=dtype)
    for name in variables:
        planes = np.empty((nz, ny, nx), dtype=np.float32)
        for k in range(nz):
            planes[k] = next(recs).reshape(ny_file, nx)[lo:hi, :]
        if name in ("bx", "by", "bz"):
            comps[name] = planes.transpose(2, 1, 0)      # -> (nx, ny, nz)
    axes = [o + s * np.arange(m) for o, s, m in zip(origin, spacing, (nx, ny, nz))]
    return GriddedField(*axes, comps["bx"], comps["by"], comps["bz"],
                        metadata={"source": path})
```

Variants you will meet:

- **One record per variable** (whole 3D array per record):
  `rec.reshape(nz, ny, nx).transpose(2, 1, 0)`.
- **One record per (variable, plane)** as above, or per (plane, variable)
  — swap the loop order.
- **8-byte markers** (`-frecord-marker=8`, some Intel builds): pass
  `marker_dtype=">i8"` (or `"<i8"`).
- **A header record** (grid sizes, time): read it first with
  `read_fortran_records(path, dtype=">i4", count=1)` and decode; then use
  `skip=1` or the iterator for the rest.
- **Discovering the layout:** the first record's length divided by the
  item size is the number of values per record. Match it against
  `nx*ny`, `nx*ny*nz`, `(nx+2g)*(ny+2g)`, ... to find plane/volume records
  and ghost widths; the total number of records then gives the variable
  count.

### Recipe D — Per-rank (domain-decomposed) chunk files

MPI codes often write one file per rank: `run.data.0000`, `.0001`, ... Each
holds a sub-block, usually including ghost layers that overlap with its
neighbours. Assembling the global grid needs the decomposition
(`px × py × pz` ranks), the block size, and the ghost width; these come from
the code's parameter file or its source. The pattern is always the same:

```python
import numpy as np
from mageometry import GriddedField
from mageometry.io import read_fortran_records

def load_chunks(pattern, n_ranks, decomposition, block, ghost, dtype=">f4",
                variables=("rho", "vx", "vy", "vz", "p", "bx", "by", "bz"),
                origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0)):
    """
    pattern: e.g. "run.data.{:04d}"; decomposition: (px, py, pz) ranks;
    block: interior size (bx_, by_, bz_) of one rank's block;
    ghost: ghost width g on each side (block on disk is interior + 2g).
    Rank r = ix + px * (iy + py * iz)  <-- check against your code!
    """
    px, py, pz = decomposition
    nx, ny, nz = (b * p for b, p in zip(block, (px, py, pz)))
    full = {v: np.empty((nx, ny, nz), np.float32) for v in ("bx", "by", "bz")}
    shape_disk = tuple(b + 2 * ghost for b in block)          # (bx_, by_, bz_)
    for r in range(n_ranks):
        ix, rem = r % px, r // px
        iy, iz = rem % py, rem // py
        recs = read_fortran_records(pattern.format(r), dtype=dtype)
        for name, rec in zip(variables, recs):
            if name not in full:
                continue
            # block stored (z, y, x) on disk -> (x, y, z); strip ghosts
            blk = rec.reshape(shape_disk[::-1]).transpose(2, 1, 0)
            g = ghost
            core = blk[g:g + block[0], g:g + block[1], g:g + block[2]]
            full[name][ix * block[0]:(ix + 1) * block[0],
                       iy * block[1]:(iy + 1) * block[1],
                       iz * block[2]:(iz + 1) * block[2]] = core
    axes = [o + s * np.arange(m) for o, s, m in zip(origin, spacing, (nx, ny, nz))]
    return GriddedField(*axes, full["bx"], full["by"], full["bz"],
                        metadata={"source": pattern, "ranks": n_ranks})
```

If you also have a merged/global file from the same run (many codes ship a
"gather" tool), assemble a few ranks and compare against it: a wrong rank
ordering or ghost width shows up as block-shaped discontinuities, and
`divergence()` lights up along the block boundaries.

### Recipe E — VTK (`.vti`, `.vtr`, `.vts`)

This recipe ships as `mageometry.io.load_vtk` (with `region`/`stride`
applied in memory after the read, and `name=('bx', 'by', 'bz')` accepted
for three scalar arrays); the code below stays as a reference for writing
your own variant. Use `pyvista` (or `vtk` directly). `ImageData` and
`RectilinearGrid` map onto `GriddedField` one-to-one; VTK stores point data
in x-fastest order.

```python
import numpy as np
import pyvista as pv
from mageometry import GriddedField

def load_vtk(path, name="B"):
    mesh = pv.read(path)                     # ImageData or RectilinearGrid
    nx, ny, nz = mesh.dimensions             # point counts
    if isinstance(mesh, pv.ImageData):
        axes = [o + s * np.arange(n) for o, s, n in zip(mesh.origin, mesh.spacing, mesh.dimensions)]
    else:                                    # RectilinearGrid: explicit axes
        axes = [np.asarray(mesh.x), np.asarray(mesh.y), np.asarray(mesh.z)]
    if name in mesh.point_data:
        B = np.asarray(mesh.point_data[name])            # (n_points, 3), x fastest
        comps = [B[:, k].reshape((nz, ny, nx)).transpose(2, 1, 0) for k in range(3)]
    else:                                                # cell data -> cell centers
        B = np.asarray(mesh.cell_data[name])
        axes = [0.5 * (a[1:] + a[:-1]) for a in axes]
        comps = [B[:, k].reshape((nz - 1, ny - 1, nx - 1)).transpose(2, 1, 0) for k in range(3)]
    return GriddedField(*axes, *comps, metadata={"source": path})
```

`StructuredGrid` (curvilinear) and unstructured meshes have no rectilinear
axes; see [Curvilinear, AMR, unstructured](#curvilinear-amr-and-unstructured-meshes).

### Recipe F — NetCDF / HDF5 / Zarr with your own layout

Use `netCDF4`/`xarray`/`h5py` to get arrays, then the same two questions
(layout, location). With `xarray` the dimension names tell you the order:

```python
import numpy as np
import xarray as xr
from mageometry import GriddedField

def load_netcdf(path, bx="Bx", by="By", bz="Bz", dims=("x", "y", "z")):
    ds = xr.open_dataset(path)
    axes = [np.asarray(ds[d].values, dtype=float) for d in dims]
    comps = [np.asarray(ds[v].transpose(*dims).values) for v in (bx, by, bz)]
    return GriddedField(*axes, *comps, metadata={"source": path, **ds.attrs})
```

`transpose(*dims)` makes the array order `(x, y, z)` regardless of how the
file stores it. For h5py, read `f[name][()]` (or a hyperslab
`f[name][zsl, ysl, xsl]` for a sub-box — see `region_slices`) and transpose.

### Recipe G — Time series from per-step files

Do not load every step; wrap your loader in a lazy `FieldSeries`:

```python
import glob, re
from mageometry.io import FieldSeries

paths = sorted(glob.glob("run/step_*.bin"))
times = [float(re.search(r"step_(\d+)", p).group(1)) * dt_output for p in paths]
series = FieldSeries.from_files(paths, load_raw, times=times,
                                nx=600, ny=400, nz=400, dtype=">f4")   # kwargs go to load_raw
series.times          # array
grid = series.at(1200.0)          # loads one step
for grid in series[::10]:         # every 10th step, one at a time
    ...
```

The bundled `load_xdmf_series` returns the same kind of object for XDMF
data.

### Cell-centered and staggered grids

- **Cell-centered** (finite-volume codes): the values belong to cell
  centers. Build the axes as `x_c = x_edges[:-1] + dx/2` (or
  `origin + (i + ½)·dx`) and pass them; nothing else changes.
- **Staggered / Yee** (constrained-transport MHD): `bx` lives on x-faces,
  `by` on y-faces, `bz` on z-faces, so the three arrays differ in shape by
  one along their own axis. Bring them to cell centers by averaging the two
  faces of each cell:

  ```python
  bx_c = 0.5 * (bx_face[1:, :, :] + bx_face[:-1, :, :])   # shape -> (nx, ny, nz)
  by_c = 0.5 * (by_face[:, 1:, :] + by_face[:, :-1, :])
  bz_c = 0.5 * (bz_face[:, :, 1:] + bz_face[:, :, :-1])
  grid = GriddedField(x_c, y_c, z_c, bx_c, by_c, bz_c)
  ```

  This is second-order accurate and is what most visualization tools do.
  (The face-averaged field is no longer exactly divergence-free to the
  discrete operator, but it is to second order — `divergence()` will still
  be small.)

### Non-uniform (stretched) axes

Supported directly — pass the actual coordinate arrays. Interpolation
(`RegularGridInterpolator`) handles arbitrary monotonic axes. Two cautions:

- The finite-difference step `delta` of the geometry functions is a
  *length*; choose it relative to the local cell size in the region you
  analyze (roughly one cell for linear interpolation).
- `stride` in `subvolume`/readers subsamples nodes, not lengths.

### Curvilinear, AMR, and unstructured meshes

`GriddedField` needs rectilinear axes. For anything else, **resample onto a
rectilinear grid** covering your region of interest, using the tools of the
code that produced the data (yt, ParaView "Resample To Image", the code's
own interpolation) or `scipy.interpolate.griddata` for modest sizes. Choose
a spacing comparable to the finest cells in the region you care about, then
proceed as in Recipe A. Keep in mind that resampling changes the data: it
smooths, and it can introduce interpolation-induced divergence — check the
divergence level of the resampled grid before trusting derived quantities.

### Units and coordinates

Do the conversion when you construct the `GriddedField`, and record it in
`metadata` so you do not do it twice:

```python
Re_km = 6371.2
grid = GriddedField(x_km / Re_km, y_km / Re_km, z_km / Re_km,
                    bx_code * B0_nT, by_code * B0_nT, bz_code * B0_nT,
                    metadata={"length_unit": "Re", "field_unit": "nT"})
```

Frame conversions: if the code's x axis points *anti*-sunward (many tail
codes) and you want GSM-like orientation, flip both the axis and the
component consistently, and keep the frame right-handed (flip two axes, or
flip one axis and one other, never a single one):

```python
# code frame: x anti-sunward, y dawnward -> flip x and y (right-handed)
x_new = -x[::-1]; y_new = -y[::-1]
bx_new = -bx[::-1, ::-1, :]; by_new = -by[::-1, ::-1, :]; bz_new = bz[::-1, ::-1, :]
```

(Reversing the arrays keeps the axes increasing.) A single-axis flip turns
the frame left-handed: the geometry still runs, but the binormal and the
sign of the torsion flip — `divergence()` does **not** catch this, so
verify handedness against something you know (dipole orientation, the
direction of the planet's field at a known point).

### Writing a proper `load_<format>()`

Template — keep all parsing inside, return a `GriddedField`, put
provenance in `metadata`:

```python
def load_myformat(path, *, region=None, stride=1, **layout):
    axes, bx, by, bz = _parse(path, **layout)          # your code
    if region is not None or stride != 1:               # optional, cheap
        from mageometry.io import region_slices
        sx, sy, sz = region_slices(axes, region, stride)
        axes = (axes[0][sx], axes[1][sy], axes[2][sz])
        bx, by, bz = bx[sx, sy, sz], by[sx, sy, sz], bz[sx, sy, sz]
    return GriddedField(*axes, bx, by, bz,
                        metadata={"source": path, "reader": "load_myformat", **layout})
```

If your reader is generally useful (a public code's output format), it fits
in `mageometry/io/` next to `xdmf.py` — same contract, plus a test that
writes a small synthetic file and reads it back.

---

## Part III — Bundled readers: XDMF + HDF5

### `load_xdmf(path, components=('BX','BY','BZ'), h5_file=None, region=None, stride=1)`

The format written by many MHD codes and readable by ParaView/VisIt: an XML
file (`.xmf`) describing the grid, pointing at heavy data in HDF5. Accepted
subset:

- One `<Grid>` with `<Topology TopologyType="3DCORECTMesh"
  NumberOfElements="NZ NY NX"/>` — a **uniform structured grid**. Other
  topologies (rectilinear with explicit axes, curvilinear, AMR,
  unstructured) are rejected; use Part II.
- `<Geometry GeometryType="ORIGIN_DXDYDZ">` with `DataItem`s named
  `Origin` and `Spacing` (or `DxDyDz`), three numbers each.
- One scalar `<Attribute>` per component, all with the same `Center`:
  `Node` (values at nodes) or `Cell` (values at cell centers,
  `origin + (i + ½)·spacing`, one fewer per axis than the topology
  declares). Names default to `BX`, `BY`, `BZ`.
  `metadata['center']` records what was found.
- `DataItem` with `Format="HDF"` referencing `file.h5:/dataset`.
- Optional `<Time Value="..."/>` → `metadata['time']`.

**Axis order:** XDMF lists dimensions and Origin/Spacing **slowest axis
first** (`NZ NY NX`, `Z0 Y0 X0`, `DZ DY DX`), and the HDF5 datasets are
C-order `(NZ, NY, NX)`. `load_xdmf` transposes to `(NX, NY, NZ)`.

**Data types:** float32/float64, either byte order (big-endian converted on
load; float32 stays float32 — 600×400×400 × 3 components ≈ 1.1 GB).

**Renamed heavy data:** `load_xdmf("run.xmf", h5_file="/archive/run-heavy.h5")`.

**Time series:** `load_xdmf` refuses temporal collections; use
`load_xdmf_series`.

### `load_xdmf_series(path, ...)`

Opens a series **lazily**. Two layouts:

- ParaView **`.xmf.series`** JSON index next to one single-grid `.xmf` per
  step:

  ```json
  {"file-series-version": "1.0",
   "files": [{"name": "run_000.xmf", "time": 0.0},
             {"name": "run_001.xmf", "time": 10.0}]}
  ```

- An **XDMF temporal collection**: one `.xmf` whose
  `<Grid GridType="Collection" CollectionType="Temporal">` holds one uniform
  `<Grid>` per step, each with `<Time Value="..."/>`.

```python
series = load_xdmf_series("run.xmf.series", region=((-15, -3), (-5, 5), (-5, 5)))
series.times; series[3]; series.at(25.0); series[::5]
```

Reader options apply to every step.

### `load_hdf5(path, datasets, origin, spacing, zyx_order=True, region=None, stride=1)`

For HDF5 datasets without XDMF metadata; you supply the geometry:

```python
grid = load_hdf5("run.h5", datasets=("BX", "BY", "BZ"),
                 origin=(x0, y0, z0), spacing=(dx, dy, dz), zyx_order=True)
```

### Reading part of a large grid

All readers take `region=((xmin, xmax), (ymin, ymax), (zmin, zmax))` (grid
coordinates, inclusive; `None` per axis = full) and `stride`. The selection
is an **HDF5 hyperslab**, so only those nodes are read:

```python
tail = load_xdmf("run.xmf", region=((-30, -5), (-10, 10), (-5, 5)))
coarse = load_xdmf("run.xmf", stride=4)
```

`GriddedField.subvolume(region, stride)` does the same in memory.

### Writing XDMF + HDF5 yourself

If you would rather convert once than write a reader, this writer produces
files `load_xdmf` accepts (it is the recipe the test suite uses):

```python
import h5py, numpy as np

# bx, by, bz: shape (nx, ny, nz); x, y, z: uniform axes
nx, ny, nz = bx.shape
origin = (x[0], y[0], z[0]); spacing = (x[1]-x[0], y[1]-y[0], z[1]-z[0])
with h5py.File("myrun.h5", "w") as f:
    for name, arr in (("BX", bx), ("BY", by), ("BZ", bz)):
        f.create_dataset(name, data=arr.transpose(2, 1, 0))     # -> (nz, ny, nx)
with open("myrun.xmf", "w") as f:
    f.write(f"""<?xml version="1.0" ?>
<Xdmf Version="2.0"><Domain>
<Grid Name="Structured Grid" GridType="Uniform">
<Topology TopologyType="3DCORECTMesh" NumberOfElements="{nz} {ny} {nx}"/>
<Geometry GeometryType="ORIGIN_DXDYDZ">
<DataItem Name="Origin" Dimensions="3" NumberType="Float" Format="XML">{origin[2]} {origin[1]} {origin[0]}</DataItem>
<DataItem Name="Spacing" Dimensions="3" NumberType="Float" Format="XML">{spacing[2]} {spacing[1]} {spacing[0]}</DataItem>
</Geometry>""")
    for name in ("BX", "BY", "BZ"):
        f.write(f"""
<Attribute Name="{name}" AttributeType="Scalar" Center="Node">
<DataItem Dimensions="{nz} {ny} {nx}" NumberType="Float" Precision="4" Format="HDF">myrun.h5:/{name}</DataItem>
</Attribute>""")
    f.write("\n</Grid></Domain></Xdmf>\n")
```

---

## Part IV — Validating a new reader

Run these on the first file you load with a new reader. Together they catch
essentially every assembly mistake.

**1. Shape and axes.** `print(grid)` shows `(nx, ny, nz)` and the axis
ranges. Do they match what you expect from the run (domain extent, cell
count per axis)? Note that a transposed load can have a *plausible* shape
when two axes have equal length — do not stop here.

**2. Divergence (the axis/component detector).**

```python
d = grid.divergence()                 # |div B| h / |B|, dimensionless
print(np.nanmedian(d), np.nanpercentile(d, 90))
```

For correctly assembled MHD output the median is ~1e-3–1e-2 (finite
differences of a discretely divergence-free field). Transposed axes,
permuted components (`by` where `bx` should be), or a sign-flipped
component give ~0.1 or more. A wrong grid *spacing* also raises it but less
dramatically. Exclude the planet/inner boundary and the outer few cells
when looking at the statistics; those are legitimately noisy.

**3. Where is the planet?** If the run contains a planet with a dipole,
`np.argmax` of `|B|` on a coarse `subvolume(stride=4)` should land at the
expected grid position; the `bz` sign near the equator tells you the
dipole orientation (for Earth-like dipoles the equatorial field points
south, `bz < 0` at `z = 0` in GSM-like frames).

**4. Physics you know.** In the inner dipole-dominated region the
equatorial curvature is `3/r` (with `r` from the planet center). Sample a
ring of points, evaluate `field_line_curvature`, and check `kappa * r / 3 ≈ 1`.
Deviations of a few percent are fine at moderate resolution; a factor of 2
or a wrong trend means a unit or spacing error.

**5. Handedness.** `divergence()` cannot see a single-axis flip. Check
that the field direction at a known point matches the physical frame (e.g.
the dipole field at the pole points along the dipole axis in the expected
sense) — or simply compare against the code's own visualization.

**6. Round trip against a known field (for writers and full pipelines).**
Sample a Tsyganenko field with `mageometry.geopack_field`, write it with
*your* writer, read it back with *your* reader, and compare interpolated
values, curvature, and Frenet frames to direct model evaluation. The test
suite does exactly this (`TestTsyganenkoFileRoundtrip` in
`tests/test_io_gridded_field.py`); copy it and swap in your functions.

---

## Part V — Practical guidance

### Interpolation and the finite-difference step

- `grid.field(method="linear")` is fast and memory-light, but its
  derivative is piecewise constant: with the geometry functions use
  `delta` of about **one grid cell**; smaller steps only sample the noise
  of the interpolant.
- `method="cubic"` has continuous derivatives: cleaner curvature and
  torsion, `delta` can be a fraction of a cell, but it is markedly slower
  on large grids — extract a `subvolume` for detailed regional analysis.
- `field_line_frame_quality(field, x, y, z, delta)` reports where the
  finite difference is not resolving the curvature; points above
  `orthogonality_tol` come back as NaN from the frame functions. Weakly
  curved regions (e.g. tail lobes at κ ~ 1e-3 per cell) are genuinely
  unresolvable from float32 grid data — that is information, not an error.
- Out-of-domain points return `fill_value` (default NaN), which propagates
  as NaN through all geometry results; a single `np.isfinite` mask handles
  them together with the intrinsically undefined points.

### Tracing through the data

`trace_field_lines(field, x, y, z, ...)` works directly on the
interpolating callable. Keep `fill_value=np.nan`: an undefined field ahead
terminates the line (status 3) instead of raising. Pass
`bounds=grid.bounds` to have lines that reach the grid edge reported as
boundary hits (status 1) with the last point placed exactly on the face.
Use `ds` of about one cell with linear interpolation.

### Memory

- Keep float32; `GriddedField` preserves the input dtype and stores the
  three components in one stacked array without duplication.
- Use `region`/`stride` (readers) or `np.memmap` + slicing (your reader)
  so that only the region of interest is materialized.
- Time series: `FieldSeries` loads one step at a time; do not hold steps in
  a list unless you need them simultaneously.
