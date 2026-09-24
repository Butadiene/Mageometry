# Predecessor release archive

[Current documentation](../README.md) · [Current installation](../../README.md#installation)

These notes record releases of Mageometry's predecessor,
`geopack-vectorize` / `geopack-vectorized`. Version numbers, installation
commands, import paths, compatibility statements, and benchmark figures
describe those historical releases. They are not the version or API
contract of the current Mageometry checkout.

| Release | Record |
| --- | --- |
| v2.0.0 | [Module reorganization and public exports](RELEASE_NOTES_v2.0.0.md) |
| v1.1.4 | [Tracing and documentation fixes](RELEASE_NOTES_v1.1.4.md) |
| v1.1.3 | [Directional derivatives](RELEASE_NOTES_v1.1.3.md) |
| v1.1.2 | [Fork attribution](RELEASE_NOTES_v1.1.2.md) |
| v1.1.1 | [Installation documentation](RELEASE_NOTES_v1.1.1.md) |
| v1.1.0 | [Vectorization and organization](RELEASE_NOTES_v1.1.0.md) |
| v1.0.12 | [Vectorized field models](RELEASE_NOTES_v1.0.12.md) |

For current use, install Mageometry from source and import `mageometry`.
Geometry lives in `mageometry.geometry`, and the field engine in
`mageometry.geopack`. See the [analysis guide](../geometry_analysis.md) and
[project README](../../README.md). The checkout's declared version is in
[`pyproject.toml`](../../pyproject.toml) and
[`mageometry/__init__.py`](../../mageometry/__init__.py).
