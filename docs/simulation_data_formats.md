# Simulation Data Formats (`mageometry.io`)

This document specifies exactly what data `mageometry.io` accepts, and how to
produce compatible files from your own simulation output.

## Architecture: everything lands on `GriddedField`

`mageometry.io` is built around one contract:

> A file-format reader's only job is to produce three coordinate axes and the
> three magnetic field components on that grid, and construct a
> `GriddedField`.

`GriddedField.field(method=..., fill_value=...)` then builds the
interpolating ``field(x, y, z) -> (bx, by, bz)`` callable consumed by every
geometry function in `mageometry.geometry`. If your format is not supported
by a bundled reader, you do not need to wait for one — read the data with any
tool you like and construct a `GriddedField` directly (see
[Any other format](#3-any-other-format-direct-construction)).

Supported entry points:

| Entry point | Input | Requires |
|---|---|---|
| `load_xdmf(path, ...)` | XDMF metadata + HDF5 heavy data (one time step) | `h5py` |
| `load_xdmf_series(path, ...)` | XDMF temporal collection or `.xmf.series` index (time series, lazy) | `h5py` |
| `load_hdf5(path, ...)` | plain HDF5 datasets + caller-supplied geometry | `h5py` |
| `GriddedField(x, y, z, bx, by, bz)` | NumPy arrays from anywhere | — |

All file readers accept `region=((xmin, xmax), (ymin, ymax), (zmin, zmax))`
and `stride` to read only a sub-box / every n-th node — see
[Reading part of a large grid](#reading-part-of-a-large-grid).

`h5py` is an optional dependency: `pip install h5py` (or
`pip install -e .[io]`).

## Units and coordinates

Mageometry does **not** interpret units. Positions are in the grid's own
length unit and field values in the grid's own field unit; geometry results
follow (curvature and directional derivatives in 1/length-unit). If you need
physical units (e.g. Re and nT), rescale the axes and/or field arrays before
constructing the `GriddedField`. The coordinate system is likewise yours —
the geometry functions only assume a right-handed Cartesian grid.

## 1. XDMF + HDF5 (`load_xdmf`)

The format written by many MHD codes and readable by ParaView/VisIt: a small
XML file (`.xmf`) describing the grid, pointing at "heavy data" in HDF5.

### Accepted XDMF subset

- One `<Grid>` with:
  - `<Topology TopologyType="3DCORECTMesh" NumberOfElements="NZ NY NX"/>` —
    a **uniform structured grid**. Other topologies (rectilinear with
    explicit axes, curvilinear, AMR, unstructured) are rejected with an
    error; use direct construction for those.
  - `<Geometry GeometryType="ORIGIN_DXDYDZ">` with two `DataItem`s named
    `Origin` and `Spacing` (or `DxDyDz`), each holding three numbers.
- One scalar `<Attribute>` per field component, all with the same
  `Center` — either `Node` (values at the grid nodes) or `Cell` (values at
  cell centers, i.e. at origin + (i + ½)·spacing, with one fewer value per
  axis than the topology declares). Default names `BX`, `BY`, `BZ`; override
  with `components=('bx', 'by', 'bz')` etc. `GriddedField.metadata['center']`
  records which convention was found.
- An optional `<Time Value="..."/>` inside the grid ends up in
  `GriddedField.metadata['time']`.
- Each attribute's `DataItem` must have `Format="HDF"` and reference the
  heavy data as `file.h5:/dataset`.

### Axis-order convention (important)

XDMF lists dimensions and Origin/Spacing values **slowest axis first**:
`NumberOfElements="NZ NY NX"`, `Origin="Z0 Y0 X0"`,
`Spacing="DZ DY DX"`. The HDF5 datasets are correspondingly C-order arrays
of shape `(NZ, NY, NX)`. `load_xdmf` transposes everything to the Mageometry
convention `(NX, NY, NZ)` with axes in (x, y, z) order — you do not have to
do anything, but your writer must follow the XDMF convention.

### Data types

float32 or float64, either endianness (big-endian files are converted to
native on load). float32 data stays float32 in memory — three components of
a 600×400×400 grid occupy ~1.1 GB.

### Renamed or relocated heavy data

The XDMF references the HDF5 by the file name it had at write time. If the
file was renamed or moved, pass the real path explicitly:

```python
grid = load_xdmf("run000.xmf", h5_file="/data/archive/run000-heavy.h5")
```

### Time series

`load_xdmf_series(path, ...)` opens a series **lazily** — nothing is read
until a step is accessed, so a long run of large grids never has to fit in
memory. Two layouts are recognized:

- a ParaView **`.xmf.series`** JSON index next to one single-grid `.xmf`
  per step:

  ```json
  {"file-series-version": "1.0",
   "files": [{"name": "run_000.xmf", "time": 0.0},
             {"name": "run_001.xmf", "time": 10.0}]}
  ```

- an **XDMF temporal collection**: one `.xmf` whose
  `<Grid GridType="Collection" CollectionType="Temporal">` holds one uniform
  `<Grid>` per step, each with `<Time Value="..."/>`.

```python
from mageometry import load_xdmf_series

series = load_xdmf_series("run.xmf.series", region=((-15, -3), (-5, 5), (-5, 5)))
series.times          # array of time values
grid = series[3]      # GriddedField for step 3 (read now)
grid = series.at(25.0)  # step closest to t = 25
for grid in series:   # one step at a time
    ...
```

Reader options (`components`, `region`, `stride`, `metadata`) apply to every
step. `load_xdmf` on a collection file raises with a pointer to
`load_xdmf_series`.

### Not (yet) supported

- Spatial collections / multi-block grids, `xi:include`, and non-HDF heavy
  data (`Format="XML"` or `Binary`): read the data yourself and construct a
  `GriddedField`.

## 2. Plain HDF5 (`load_hdf5`)

When you have the HDF5 datasets but no XDMF metadata, supply the grid
geometry yourself:

```python
from mageometry import load_hdf5

grid = load_hdf5(
    "run000.h5",
    datasets=("BX", "BY", "BZ"),   # dataset names in the file
    origin=(x0, y0, z0),           # coordinate of the first node, (x, y, z)
    spacing=(dx, dy, dz),          # grid step, (x, y, z)
    zyx_order=True,                # datasets stored (NZ, NY, NX)? (default)
)
```

Set `zyx_order=False` if your datasets are already stored `(NX, NY, NZ)`.

## Reading part of a large grid

Full MHD grids (hundreds of nodes per axis, float32) are large; often only a
region is of interest. Every file reader takes

- `region=((xmin, xmax), (ymin, ymax), (zmin, zmax))` in grid coordinates
  (inclusive bounds; `None` per axis keeps the full extent), and
- `stride=n` or `stride=(nx, ny, nz)` to keep every n-th node.

The selection is applied as an **HDF5 hyperslab**, so only the selected
nodes are ever read from disk:

```python
tail = load_xdmf("run000.xmf", region=((-30, -5), (-10, 10), (-5, 5)))
coarse = load_xdmf("run000.xmf", stride=4)     # quick look at the whole domain
```

For data already in memory, `GriddedField.subvolume(region, stride)` returns
the same kind of sub-grid (as a copy).

## 3. Any other format: direct construction

`GriddedField` accepts arrays from anywhere (VTK, NetCDF, CDF, raw Fortran
binaries, in-memory simulation state, ...):

```python
import numpy as np
from mageometry import GriddedField

# x, y, z: 1D, strictly increasing. Uniform spacing is NOT required here —
# any rectilinear grid works (only the file readers assume uniform grids).
# bx, by, bz: shape (len(x), len(y), len(z))
grid = GriddedField(x, y, z, bx, by, bz, metadata={"source": "my_run_42"})
field = grid.field(method="linear", fill_value=np.nan)
```

Requirements checked at construction:

- axes are 1D, at least 2 points, strictly increasing;
- each component has shape `(len(x), len(y), len(z))`.

To support a new file format properly, write a small
`load_<format>(path, ...) -> GriddedField` function following
`mageometry/io/xdmf.py` as a template, and keep all parsing inside it.

## Creating compatible files from your own simulation

Minimal writer producing a valid XDMF + HDF5 pair (this is also how the
reader tests generate their fixtures):

```python
import h5py
import numpy as np

# bx, by, bz: your arrays, shape (nx, ny, nz), any float dtype
nx, ny, nz = bx.shape
origin = (x0, y0, z0)
spacing = (dx, dy, dz)

with h5py.File("myrun.h5", "w") as f:
    for name, arr in (("BX", bx), ("BY", by), ("BZ", bz)):
        f.create_dataset(name, data=arr.transpose(2, 1, 0))  # -> (nz, ny, nx)

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

Round-trip check: `load_xdmf("myrun.xmf")` must return a `GriddedField`
whose `.bx` etc. equal your original `(nx, ny, nz)` arrays.

The test suite validates this pipeline against an exactly reproducible
reference: `TestTsyganenkoFileRoundtrip` in `tests/test_io_gridded_field.py`
samples a T96+dipole model field onto a grid, writes it with exactly the
recipe above, loads it back, and checks interpolated field values, curvature,
and Frenet frames against direct model evaluation. Use it as a template for
verifying your own writer.

## Interpolation guidance for geometry analysis

- The geometry functions differentiate the field by finite differences with
  step `delta`. Under `method='linear'` the interpolant's derivative is
  piecewise constant, which makes curvature/torsion noisy for `delta` much
  smaller than the grid spacing. Rule of thumb: use `delta` of about one
  grid step with linear interpolation.
- `method='cubic'` gives smooth derivatives and cleaner geometry, but is
  substantially slower on large grids; consider extracting a subvolume
  (slice the arrays before constructing the `GriddedField`) for detailed
  regional analysis.
- Out-of-domain points return `fill_value` (default NaN), which propagates
  into geometry results as NaN — the same convention the geometry functions
  use for degenerate points (zero field, straight lines, unresolved
  curvature), so a single `np.isfinite` mask covers both.
- Keep large data as float32; `GriddedField` preserves the input dtype and
  stores the three components in one stacked array without duplication.

## Tracing field lines through the data

`mageometry.trace_field_lines(field, x, y, z, ...)` works on the interpolating
callable directly. Keep the default `fill_value=np.nan`: the tracer treats an
undefined (non-finite or zero) field ahead as a termination condition, so
lines end at the domain edge instead of raising. Pass `bounds=grid.bounds`
to have such lines reported as boundary hits (`STATUS_OUTER`, last point
placed exactly on the box face) rather than `STATUS_INVALID`; lines that
run into a zero-field region inside the box (e.g. a masked planet core) stay
`STATUS_INVALID`. Use a step `ds` of about one grid cell with linear
interpolation; the traced points can be fed straight into the geometry
functions.
