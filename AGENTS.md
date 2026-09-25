# Repository Guidelines

## Project Structure & Module Organization

Production code lives in `mageometry/`. Treat `mageometry.geometry` as the primary analysis API; `geopack/` contains scalar and vectorized magnetic-field models, `io/` handles gridded simulation data, `tracing.py` provides generic field-line tracing, and `viz/` and `viz3d/` contain optional plotting features. Keep tests in the flat `tests/test_*.py` suite. Runnable material belongs in `examples/`, benchmark and README-regeneration scripts in `benchmark/`, and user documentation in `docs/`. Files under `mageometry/geopack/igrf_coeffs/` are packaged runtime assets.

## Build, Test, and Development Commands

```bash
python -m pip install -e '.[dev]'       # editable install with test/optional dependencies
python -m unittest discover tests/      # run the complete test suite
python tests/test_vectorized_models.py  # run one test file
python -m build                         # build wheel and source distribution locally
```

Install `.[examples]` instead when working on notebooks. The package is private and is not published to PyPI; keep the `Private :: Do Not Upload` classifier intact.

## Coding Style & Naming Conventions

Target Python 3.9+ and use four-space indentation. Follow `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants. No formatter or linter is configured, so match nearby code and keep imports and functions readable. Write comments, docstrings, and documentation in English; public scientific APIs use NumPy-style `Parameters` and `Returns` sections. Vectorized code should preserve NumPy broadcasting and scalar-in/scalar-out behavior, guard division safely, and return `NaN` for undefined geometry rather than zero sentinels.

## Scientific Notation and Theory

Use [docs/fac_anisotropy_theory.md](docs/fac_anisotropy_theory.md) as the
baseline for FAC/transverse-geometry notation and interpretation. Use
`beta_g` and `delta_g` consistently in numerical variables, public APIs,
viewers, CLIs, tests, and documentation, without alternative diagnostic
names. The symbol q denotes only a transverse matrix entry. Keep the
theory self-contained and distinguish mathematical identities, assumptions,
and proposed R1/R2 research hypotheses.

## Testing Guidelines

Tests use `unittest`: name files `test_<feature>.py`, classes `Test*`, and methods `test_*`. Use `np.testing.assert_allclose` for numerical comparisons and `skipUnless` for optional dependencies. Add focused regression coverage beside every behavior change. There is no configured coverage threshold. Field-model tolerances can be adjusted with `GEOPACK_FIELD_RTOL`, `GEOPACK_FIELD_ATOL`, and `GEOPACK_MAXULP`.

## Commit & Pull Request Guidelines

Use concise, capitalized imperative subjects such as `Add gridded-field validation`; Conventional Commit prefixes are not used. Keep commits focused. Pull requests should summarize behavior and tradeoffs, report commands run, and update documentation, examples, or `CHANGELOG.md` when public behavior changes. Link relevant issues and include screenshots for visualization changes.

## Configuration & Safety

Never commit local settings or large simulation artifacts (`*.h5`, `*.data*`, `*.xmf`, archives). Importing `mageometry` must not access the network. For releases, keep versions synchronized in `pyproject.toml` and `mageometry/__init__.py`.
