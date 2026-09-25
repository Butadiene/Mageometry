# Simulation Data: Formats and Bringing Your Own (`mageometry.io`)

[Documentation index](README.md) · [Geometry analysis](geometry_analysis.md)
· [Viewer guide](viewer.md)

Convert your magnetic-field output into a `GriddedField`, then use the same
analysis API as for analytic fields. Start with a bundled reader when its
format matches your data; otherwise adapt a recipe to your file layout.
All filenames, array names, and sample layouts in this guide are placeholders.

| Your input | Start here | Optional dependency |
| --- | --- | --- |
| Uniform XDMF + HDF5 | [Bundled XDMF reader](#xdmf-snapshots) | `.[io]` |
| Plain HDF5 with known origin/spacing | [Direct HDF5](#direct-hdf5) | `.[io]` |
| VTK ImageData / RectilinearGrid | [VTK recipe](#recipe-e--vtk-vti-vtr) | `.[viz3d]` |
| NumPy / raw binary / Fortran records | [Reader cookbook](#part-ii--cookbook-from-your-files-to-griddedfield) | Base NumPy/SciPy install |
| Time series | [Series reader](#xdmf-time-series) or [custom series](#recipe-g--time-series-from-per-step-files) | Depends on the underlying reader |

Install extras from the repository root, for example
`python -m pip install -e '.[io,viz3d]'`. For file selection and display
controls, use the [viewer guide](viewer.md).

Part I describes the one data structure everything lands on. Part II is a
cookbook for the formats simulation codes actually produce. Part III covers
the bundled XDMF/HDF5 readers. Part IV is the validation checklist you
should run on any newly written reader, and Part V has practical guidance
on interpolation, tracing, and memory.

---

## Part I — The one contract: `GriddedField`

```python
import numpy as np
from mageometry import GriddedField

# A small synthetic example; substitute your axes and components here.
x = np.linspace(-2, 2, 17)
y = np.linspace(-3, 3, 19)
z = np.linspace(-1, 1, 13)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
bx, by, bz = -Y, X, np.ones_like(Z)
grid = GriddedField(x, y, z, bx, by, bz, metadata={"source": "synthetic"})
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
| `metadata` | Optional dict, shallow-copied (provenance, unit labels, ...); values are not interpreted. |

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
  input stays float32 when all components use that dtype. The stored dtype
  is chosen from `bx` and float32; convert all three components to a common
  float dtype first when their precisions differ.

Building blocks provided for your reader (all in `mageometry.io`):

| Tool | Purpose |
|---|---|
| `read_fortran_records(path, dtype)` / `iter_fortran_records` | Fortran unformatted sequential files (record-length markers) |
| `region_slices(axes, region, stride)` | Turn a bounding box / stride into index slices |
| `GriddedField.subvolume(region, stride)` | Cut a sub-box out of an in-memory grid |
| `GriddedField.divergence()` | Numerical divergence diagnostic; can reveal layout errors, but does not prove correctness |
| `FieldSeries.from_files(paths, loader, times)` | Lazy time series from one file per step and your loader |

---

## Part II — Cookbook: from your files to `GriddedField`

The reader functions below handle their stated example layouts. Adapt
headers, variables, rank ordering, and centering to your format. Other
snippets illustrate transformations or depend on caller-supplied values;
they are labelled accordingly. Third-party packages used in a recipe,
such as xarray and its file backend, are not installed by Mageometry.

### The mental model: three questions

Before writing any code, answer these for your data; every recipe below is
an instance of them.

1. **What is the memory layout of one component?** Is the fastest-varying
   index x or z? C-order `(nz, ny, nx)` and Fortran-order `(nx, ny, nz)`
   both have x varying fastest; C-order `(nx, ny, nz)` has z fastest.
   Are there ghost/guard cells to strip? Any header bytes?
2. **Where are the values located?** Node positions (`origin + i·dx`) or
   cell centers (`origin + (i + ½)·dx`)? Uniform or stretched axes? Are the
   three components at the same locations?
3. **What are the units and the frame?** Code units → physical units;
   which axis is which; is the frame right-handed?

Check the loaded arrays against known values as well as their divergence.
Some ordering errors preserve zero divergence; see
[Part IV](#part-iv--validating-a-new-reader).

### Recipe A — NumPy arrays already in memory / `.npy` / `.npz`

```python
import numpy as np
from mageometry import GriddedField

def load_npz(path):
    # Adjust the key names and the axis order to your file. Here the
    # arrays are stored (nz, ny, nx).
    with np.load(path) as d:
        bx, by, bz = (d[k].transpose(2, 1, 0) for k in ("bx", "by", "bz"))
        return GriddedField(d["x"], d["y"], d["z"], bx, by, bz,
                            metadata={"source": str(path)})
```

`transpose(2, 1, 0)` is a view — no copy until `GriddedField` stacks the
components.

### Recipe B — Raw binary (`np.fromfile` / `np.memmap`)

For raw dumps, specify the element type, byte order, header size, and layout.

```python
import numpy as np
from mageometry import GriddedField

def load_raw(path, nx, ny, nz, dtype=">f4", header_bytes=0,
             variables=("rho", "vx", "vy", "vz", "p", "bx", "by", "bz"),
             origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0)):
    """One file holding `variables` back to back, each stored (nz, ny, nx)."""
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

The helpers support positive-length records with matching leading/trailing
markers. Negative continuation markers are rejected; direct-access files
without record markers need a raw-layout reader instead.

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
    Variables are consecutive; record k belongs to variable k // nz, plane k % nz.
    Each plane is stored y-major: reshape to (ny + ghosts, nx).
    """
    ny_file = ny + sum(ny_ghost)
    lo, hi = ny_ghost[0], ny_ghost[0] + ny
    comps = {}
    recs = iter_fortran_records(path, dtype=dtype)
    for name in variables:
        planes = np.empty((nz, ny, nx), dtype=np.dtype(dtype).newbyteorder('='))
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
  `skip=1` with `read_fortran_records`, or `start=header_end_byte_offset`
  with the iterator for the rest. Skipped records are still decoded with
  the selected payload dtype, so use a byte offset for mixed record types
  that cannot all be decoded with the same dtype.
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
    if n_ranks != px * py * pz:
        raise ValueError('n_ranks must match the decomposition.')
    if not {'bx', 'by', 'bz'}.issubset(variables):
        raise ValueError('variables must include bx, by, and bz.')
    nx, ny, nz = (b * p for b, p in zip(block, (px, py, pz)))
    native_dtype = np.dtype(dtype).newbyteorder('=')
    full = {v: np.empty((nx, ny, nz), native_dtype) for v in ("bx", "by", "bz")}
    shape_disk = tuple(b + 2 * ghost for b in block)          # (bx_, by_, bz_)
    for r in range(n_ranks):
        ix, rem = r % px, r // px
        iy, iz = rem % py, rem // py
        recs = read_fortran_records(pattern.format(r), dtype=dtype)
        if len(recs) != len(variables):
            raise ValueError('Expected one record per variable in each rank file.')
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

### Recipe E — VTK (`.vti`, `.vtr`)

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
    elif isinstance(mesh, pv.RectilinearGrid):
        axes = [np.asarray(mesh.x), np.asarray(mesh.y), np.asarray(mesh.z)]
    else:
        raise ValueError('Only ImageData and RectilinearGrid are supported.')
    if name in mesh.point_data:
        B = np.asarray(mesh.point_data[name])            # (n_points, 3), x fastest
        comps = [B[:, k].reshape((nz, ny, nx)).transpose(2, 1, 0) for k in range(3)]
    else:                                                # cell data -> cell centers
        B = np.asarray(mesh.cell_data[name])
        axes = [0.5 * (a[1:] + a[:-1]) for a in axes]
        comps = [B[:, k].reshape((nz - 1, ny - 1, nx - 1)).transpose(2, 1, 0) for k in range(3)]
    return GriddedField(*axes, *comps, metadata={"source": path})
```

The bundled reader also validates array names and component counts. Use
`from mageometry import load_vtk` in normal applications; the simplified
function above illustrates point/cell ordering. `StructuredGrid` (`.vts`)
and unstructured meshes are rejected; see
[Curvilinear, AMR, unstructured](#curvilinear-amr-and-unstructured-meshes).

### Recipe F — NetCDF / HDF5 / Zarr with your own layout

Use `netCDF4`/`xarray`/`h5py` to get arrays, then the same two questions
(layout, location). With `xarray` the dimension names tell you the order:

```python
import numpy as np
import xarray as xr
from mageometry import GriddedField

def load_netcdf(path, bx="Bx", by="By", bz="Bz", dims=("x", "y", "z")):
    with xr.open_dataset(path) as ds:
        axes = [np.asarray(ds[d].values, dtype=float) for d in dims]
        comps = [np.asarray(ds[v].transpose(*dims).values) for v in (bx, by, bz)]
        return GriddedField(*axes, *comps, metadata={"source": str(path), **ds.attrs})
```

`transpose(*dims)` makes the array order `(x, y, z)` regardless of how the
file stores it. For h5py, read `f[name][()]` (or a hyperslab
`f[name][zsl, ysl, xsl]` for a sub-box — see `region_slices`) and transpose.

### Recipe G — Time series from per-step files

Do not load every step; wrap your loader in a lazy `FieldSeries`. Set
`nx`, `ny`, `nz`, `dtype`, and `dt_output` from your data's format and
output settings:

```python
import glob
from pathlib import Path
import re
from mageometry.io import FieldSeries

def step_number(path):
    return int(re.fullmatch(r"step_(\d+)\.bin", Path(path).name).group(1))

paths = sorted(glob.glob("run/step_*.bin"), key=step_number)
if not paths:
    raise ValueError('No step files matched the pattern.')
times = [step_number(path) * dt_output for path in paths]
series = FieldSeries.from_files(paths, load_raw, times=times,
                                nx=nx, ny=ny, nz=nz, dtype=dtype)   # kwargs go to load_raw
series.times          # array
grid = series.at(times[0])        # loads the first step
for grid in series[::10]:         # every 10th step, one at a time
    ...
```

The bundled `load_xdmf_series` returns the same kind of object for XDMF
data. This snippet also uses `load_raw` from Recipe B. `series.at(t)` loads
the nearest step; it does not interpolate time. Repeated indexing reloads
the step, so keep a loaded grid when reusing it.

### Cell-centered and staggered grids

- **Cell-centered** (finite-volume codes): the values belong to cell
  centers. Build the axes as `x_c = 0.5 * (x_edges[:-1] + x_edges[1:])`
  (and likewise for y/z); this also works for stretched axes.
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

  This is a centered interpolation for midpoint cells. Confirm the actual
  face/centre locations before using it on other layouts. It does not
  preserve a solver's discrete divergence constraint in general; evaluate
  the resulting field with independent checks below.

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
    from mageometry import GriddedField
    from mageometry.io import region_slices

    axes, bx, by, bz = _parse(path, **layout)          # your code
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

### XDMF snapshots

```python
from mageometry import load_xdmf

grid = load_xdmf('snapshot.xmf', components=('BX', 'BY', 'BZ'), stride=1)
```

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
load; float32 stays float32 — field storage uses approximately
`nx * ny * nz * 3 * 4` bytes for three float32 components).

**Paths and renamed heavy data:** referenced HDF5 paths are relative to the
XDMF file's directory. `load_xdmf('snapshot.xmf', h5_file='field.h5')`
overrides the heavy file while retaining the referenced dataset paths.
A relative override is resolved from the working directory.

**Time series:** `load_xdmf` refuses temporal collections; use
`load_xdmf_series`.

### XDMF time series

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
from mageometry import load_xdmf_series

series = load_xdmf_series('snapshots.xmf.series')
print(series.times)
grid = series[0]                 # reads one step
every_fifth_step = series[::5]    # still lazy
```

`components`, `region`, `stride`, and `metadata` apply to every step.
This API has no `h5_file` override. `series.at(t)` selects the nearest known
time without interpolating; index by step when times are unavailable.
Unknown times are NaN. A new access reloads a step rather than caching it.

### Direct HDF5

For HDF5 datasets without XDMF metadata, supply the geometry in x/y/z order.
The values below describe an example layout; replace them with yours:

```python
from mageometry import load_hdf5

grid = load_hdf5('field.h5', datasets=('BX', 'BY', 'BZ'),
                 origin=(0, 0, 0), spacing=(1, 1, 1), zyx_order=True)
```

`zyx_order=True` reads `(nz, ny, nx)` arrays; `False` reads `(nx, ny, nz)`.
The API defaults to zero origin and unit spacing. The CLI requires these
values explicitly to avoid silently assigning coordinates. For cell data,
pass the coordinate of the first **cell centre** as the origin.

### Reading part of a large grid

All bundled readers take `region=((xmin, xmax), (ymin, ymax), (zmin, zmax))`
(grid coordinates, inclusive; `None` per axis = full) and `stride` (positive
integer or three integers). For XDMF/HDF5 the selection is an **HDF5
hyperslab**, so only those samples are materialized:

```python
from mageometry import load_xdmf

# Choose bounds and stride that leave enough nodes in your file.
region = load_xdmf('snapshot.xmf', region=((-1, 1), None, None))
coarse = load_xdmf('snapshot.xmf', stride=2)
```

VTK input is read in full, then subset in memory.
`GriddedField.subvolume(region, stride)` copies selected nodes from an
already loaded grid. Keep at least two points per axis for `GriddedField`,
and three for the current/geometry overview. Stride samples nodes; it does
not average cells or automatically retain the final endpoint.

### Writing XDMF + HDF5 yourself

This writer accepts a uniform `GriddedField` and an output filename. It
writes scalar node data with matching HDF5 dtype and XDMF precision. The
output geometry follows the same ZYX convention used by the reader tests.

```python
from pathlib import Path
import xml.etree.ElementTree as ET
import h5py
import numpy as np

def write_xdmf(path, grid, dtype=np.float32):
    path = Path(path)
    heavy_path = path.with_suffix('.h5')
    dtype = np.dtype(dtype)
    if dtype.kind != 'f' or dtype.itemsize not in (4, 8):
        raise ValueError('Use float32 or float64 output.')
    axes = (grid.x, grid.y, grid.z)
    spacing = tuple(axis[1] - axis[0] for axis in axes)
    if not all(np.allclose(np.diff(axis), step, rtol=1e-8, atol=0)
               for axis, step in zip(axes, spacing)):
        raise ValueError('This XDMF writer requires uniform axes.')
    root = ET.Element('Xdmf', Version='2.0')
    node = ET.SubElement(ET.SubElement(root, 'Domain'), 'Grid', GridType='Uniform')
    dimensions = ' '.join(str(n) for n in grid.shape[::-1])
    ET.SubElement(node, 'Topology', TopologyType='3DCORECTMesh',
                  NumberOfElements=dimensions)
    geometry = ET.SubElement(node, 'Geometry', GeometryType='ORIGIN_DXDYDZ')
    for name, values in (('Origin', [axis[0] for axis in axes]), ('Spacing', spacing)):
        item = ET.SubElement(geometry, 'DataItem', Name=name, Dimensions='3',
                             NumberType='Float', Format='XML')
        item.text = ' '.join(str(value) for value in values[::-1])
    with h5py.File(heavy_path, 'w') as heavy:
        for name, values in zip(('BX', 'BY', 'BZ'), (grid.bx, grid.by, grid.bz)):
            heavy.create_dataset(name, data=values.transpose(2, 1, 0), dtype=dtype)
            attribute = ET.SubElement(node, 'Attribute', Name=name,
                                       AttributeType='Scalar', Center='Node')
            item = ET.SubElement(attribute, 'DataItem', Dimensions=dimensions,
                                 NumberType='Float', Precision=str(dtype.itemsize),
                                 Format='HDF')
            item.text = f'{heavy_path.name}:/{name}'
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
```

With the synthetic grid from Part I, check the round trip. Allow a small
absolute tolerance for roundoff when reconstructing coordinates:

```python
from mageometry import load_xdmf

write_xdmf('snapshot.xmf', grid, dtype=np.float64)
restored = load_xdmf('snapshot.xmf')
for actual, expected in zip((restored.x, restored.y, restored.z), (grid.x, grid.y, grid.z)):
    np.testing.assert_allclose(actual, expected, atol=1e-12)
np.testing.assert_allclose(restored.b, grid.b)
```

---

## Part IV — Validating a new reader

Use several independent checks on the first file loaded with a new reader.
No single diagnostic proves correct layout, units, or handedness.

**1. Shape and axes.** `print(grid)` shows `(nx, ny, nz)` and the axis
ranges. Do they match what you expect from the run (domain extent, cell
count per axis)? Note that a transposed load can have a *plausible* shape
when two axes have equal length — do not stop here.

**2. Divergence (a diagnostic, not a pass/fail certificate).**

```python
import numpy as np

d = grid.divergence()                 # |div B| h / |B|, dimensionless
finite = d[np.isfinite(d)]
if finite.size:
    print(np.median(finite), np.percentile(finite, 90))
else:
    print('No finite divergence samples; inspect the input field and mask.')
```

`h` is the mean of the three axes' mean spacings; it is a global scale,
including on stretched grids. `grid.divergence(relative=False)` returns
signed divergence in field/length units. There is no universal threshold:
the result depends on resolution, boundary stencils, interpolation, and
the source field. Some wrong axis/component assignments still have zero
divergence. Compare with a reference field and spatial maps, and exclude
known invalid regions when computing statistics. The relative result is
NaN at zero field and can be large near weak fields.

**3. Where is the planet?** If the run contains a planet with a dipole,
compare field magnitude and direction around the expected centre with
the source model. Masked interiors, boundary conditions, and other current
systems can move the strongest sample away from the centre. For a dipole
moment m, the ideal equatorial field points opposite m; check the moment
and frame conventions rather than assuming a universal sign for `bz`.

**4. Physics you know.** In the inner dipole-dominated region the
equatorial curvature is `3/r` (with `r` from the planet center). Sample a
ring of points, evaluate `field_line_curvature`, and check `kappa * r / 3 ≈ 1`.
Compare errors across grid resolutions and derivative steps. Departures
can reflect interpolation, finite differences, or nondipolar physics as
well as incorrect units or spacing.

**5. Handedness.** `divergence()` cannot see a single-axis flip. Check
that the field direction at a known point matches the physical frame (e.g.
the dipole field at the pole points along the dipole axis in the expected
sense) — or simply compare against the code's own visualization.

**6. Round trip against a known field (for writers and full pipelines).**
Use unequal axis lengths and a field with distinguishable components;
compare axes and every component at nodes before testing interpolation and
geometry. The [synthetic round trip above](#writing-xdmf--hdf5-yourself)
is a starting point. Model-based checks are in `TestTsyganenkoFileRoundtrip`
in [`tests/test_io_gridded_field.py`](../tests/test_io_gridded_field.py).
Also compare against an independent file reader or known samples: a writer
and reader sharing the same ordering mistake can pass a round-trip test.

---

## Part V — Practical guidance

### Interpolation and the finite-difference step

- `grid.field(method="linear")` uses multilinear interpolation. Its first
  derivatives generally jump at cell faces; they need not be constant
  throughout a 3D cell. A step near the local grid spacing is a useful
  starting point, then compare nearby step sizes. Subcell estimates
  describe the interpolant, not unresolved structure in the source data.
- `method="cubic"` can provide smoother derivative estimates. It requires
  a SciPy version supporting that method, at least four nodes per axis,
  and finite input values. Construction may use substantially more memory
  and time; extract a finite subvolume first. Higher-order interpolation
  does not automatically improve noisy data.
- `field_line_frame_quality(field, x, y, z, delta)` reports where the
  finite difference is not resolving the curvature; points above
  `orthogonality_tol` have NaN normals/binormals. Weak curvature is sensitive
  to spacing and input precision; there is no universal float32 curvature
  cutoff. For transverse diagnostics, compare `gamma` with the
  frame-dependent `beta_g` and `delta_g`.
- Out-of-domain points return `fill_value` (default NaN), which propagates
  as NaN through dependent geometry results. Inspect validity separately
  for each quantity and leave room for neighbouring derivative stencils.
  `grid.field(fill_value=None)` raises on out-of-domain queries; this
  wrapper does not use `None` to request extrapolation.

### Tracing through the data

`trace_field_lines(field, x, y, z, ...)` works directly on the
interpolating callable. Keep `fill_value=np.nan`: an undefined field ahead
terminates the line (status 3) instead of raising. Pass
`bounds=grid.bounds` to have lines that reach the grid edge reported as
boundary hits (status 1) with the last point placed exactly on the face.
Use `ds` of about one cell with linear interpolation.

### Memory

- Choose a common float32 or float64 component dtype for your accuracy
  needs. Construction allocates one stacked array; the input arrays still
  occupy memory while referenced. `grid.bx/by/bz` are views of that array.
  Coordinates are float64 and many analysis temporaries are float64.
- Use HDF5 reader `region`/`stride`, or `np.memmap` + slicing in a custom
  reader, to materialize only the region of interest. VTK reads the full
  file first. `subvolume` allocates a new grid, and viewer `max_points`
  limits the preview after loading.
- Time series: `FieldSeries` loads one step at a time; do not hold steps in
  a list unless you need them simultaneously.

For a full workflow, continue with [geometry analysis](geometry_analysis.md)
or the [viewer guide](viewer.md).
