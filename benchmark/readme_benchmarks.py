"""
README-ready Performance Benchmarks

Produces compact Markdown tables benchmarking n=100 and n=1000 points for:
  - Coordinate transforms (aggregate of common transforms)
  - IGRF (GSW if available; else GSM)
  - T89 / T96 / T01 / T04 (external models)
  - Field line tracing (if a vectorized tracer exists; otherwise N/A)

Usage:
    python benchmark/readme_benchmarks.py           # Markdown tables (default)
    python benchmark/readme_benchmarks.py --plain   # plain text output
"""

import argparse
import gc
import time

import numpy as np
import pandas as pd

import geopack

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
N_REPEATS = 3

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
from datetime import datetime

ut = datetime(2020, 1, 1, 12, 0, 0).timestamp()
ps = geopack.recalc(ut)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now():
    return time.perf_counter()


def _as_f64(a):
    return np.asarray(a, dtype=np.float64)


def sample_xyz(n_points, xlim, ylim, zlim, seed=SEED):
    """Sample Cartesian points with float64 dtype."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(xlim[0], xlim[1], n_points)
    y = rng.uniform(ylim[0], ylim[1], n_points)
    z = rng.uniform(zlim[0], zlim[1], n_points)
    return _as_f64(x), _as_f64(y), _as_f64(z)


def _bench_exact(scalar_call, vector_call, n_points, n_repeats=N_REPEATS):
    """Time scalar and vector calls, return (t_scalar, t_vector, speedup)."""
    t_s = []
    for _ in range(n_repeats):
        t0 = _now()
        scalar_call(n_points)
        t_s.append(_now() - t0)
    t_scalar = float(np.min(t_s))

    t_v = []
    for _ in range(n_repeats):
        t0 = _now()
        vector_call(n_points)
        t_v.append(_now() - t0)
    t_vec = float(np.min(t_v))

    speedup = t_scalar / max(t_vec, 1e-15)
    return t_scalar, t_vec, speedup


def _make_xyz_for_models(n_points, model_name, seed=SEED):
    if model_name.upper() in ("T01", "T04"):
        return sample_xyz(n_points, (-15.0, 12.0), (-15, 15), (-15, 15), seed=seed)
    return sample_xyz(n_points, (-25.0, 12.0), (-15, 15), (-15, 15), seed=seed)


def _sample_trace_start_points(n_lines, rmin=1.2, rmax=8.0, seed=SEED + 21):
    """Sample start points on a spherical shell to ensure r > r0."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(rmin, rmax, n_lines)
    u = rng.uniform(-1.0, 1.0, n_lines)
    phi = rng.uniform(0.0, 2 * np.pi, n_lines)
    sin_t = np.sqrt(1.0 - u * u)
    x = r * sin_t * np.cos(phi)
    y = r * sin_t * np.sin(phi)
    z = r * u
    return _as_f64(x), _as_f64(y), _as_f64(z)


def _call_trace_scalar(x, y, z, *, dir=1.0, rlim=10.0, r0=1.0,
                       parmod=2, exname="t89", inname="igrf", maxloop=200):
    """Call scalar geopack.trace with best-effort compatibility."""
    try:
        return geopack.trace(
            float(x), float(y), float(z), dir,
            rlim=rlim, r0=r0, parmod=parmod,
            exname=exname, inname=inname, maxloop=maxloop,
        )
    except TypeError:
        return geopack.trace(
            float(x), float(y), float(z), dir,
            rlim=rlim, r0=r0, parmod=parmod,
            exname=exname, inname=inname,
        )


def _call_trace_vector(x, y, z, *, dir=1.0, rlim=10.0, r0=1.0,
                       parmod=2, exname="t89", inname="igrf",
                       maxloop=200, return_full_path=True,
                       strict_scalar_models=False):
    """Call vectorized geopack.trace_vectorized."""
    return geopack.trace_vectorized(
        x, y, z, dir=dir, rlim=rlim, r0=r0, parmod=parmod,
        exname=exname, inname=inname, maxloop=maxloop,
        return_full_path=return_full_path,
        strict_scalar_models=strict_scalar_models,
    )


def _bench_trace(n_target, *, max_scalar_lines=50,
                 n_repeats=N_REPEATS, maxloop=200,
                 return_full_path=True, strict_scalar_models=False):
    """
    Benchmark tracing n_target lines.

    Scalar: measure up to max_scalar_lines and extrapolate.
    Vector: run full n_target in one call.

    Returns (t_scalar, t_vec, speedup, extrapolated, n_scalar_measured).
    """
    xs, ys, zs = _sample_trace_start_points(n_target)

    # Scalar (cap + extrapolate)
    n_scalar = min(n_target, max_scalar_lines)
    t_s = []
    for _ in range(n_repeats):
        t0 = _now()
        for i in range(n_scalar):
            _call_trace_scalar(xs[i], ys[i], zs[i], maxloop=maxloop)
        t_s.append(_now() - t0)
    t_scalar_meas = float(np.min(t_s))
    extrapolated = n_scalar < n_target
    t_scalar = (t_scalar_meas * (n_target / max(n_scalar, 1))
                if extrapolated else t_scalar_meas)

    # Vector (full batch)
    t_v = []
    for _ in range(n_repeats):
        gc.collect()
        t0 = _now()
        out = _call_trace_vector(
            xs, ys, zs, maxloop=maxloop,
            return_full_path=return_full_path,
            strict_scalar_models=strict_scalar_models,
        )
        del out
        t_v.append(_now() - t0)
    t_vec = float(np.min(t_v))

    speedup = t_scalar / max(t_vec, 1e-15)
    return t_scalar, t_vec, speedup, extrapolated, n_scalar


def _fmt_speedup(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    if x >= 100:
        return f"**{x:.0f}x**"
    return f"**{x:.1f}x**"


# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------
parmod_default = np.zeros(10, dtype=np.float64)
parmod_default[0] = 3.0    # Pdyn (nPa)
parmod_default[1] = -20.0  # Dst (nT)
parmod_default[2] = 0.0    # ByIMF (nT)
parmod_default[3] = -5.0   # BzIMF (nT)

parmod_t01 = parmod_default.copy()
parmod_t01[4] = 2.0
parmod_t01[5] = 3.0

parmod_t04 = parmod_default.copy()
parmod_t04[4:10] = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

model_functions = [
    ("T89", geopack.t89, geopack.t89_vectorized, 4),
    ("T96", geopack.t96, geopack.t96_vectorized, parmod_default),
    ("T01", geopack.t01, geopack.t01_vectorized, parmod_t01),
    ("T04", geopack.t04, geopack.t04_vectorized, parmod_t04),
]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def run_benchmarks(n_points):
    """Run all benchmarks for the given number of points. Returns a DataFrame."""
    bench_rows = []

    # -- Coordinate transforms (aggregate) ------------------------------
    coord_subset = [
        ("gsmgse", geopack.gsmgse, geopack.gsmgse_vectorized, 1),
        ("geogsm", geopack.geogsm, geopack.geogsm_vectorized, 1),
        ("geomag", geopack.geomag, geopack.geomag_vectorized, 1),
        ("gswgsm", geopack.gswgsm, geopack.gswgsm_vectorized, 1),
    ]
    x, y, z = sample_xyz(n_points, (-15, 15), (-15, 15), (-15, 15), seed=SEED + 11)

    def _scalar_call(_n):
        for i in range(_n):
            for _, fs, _, d in coord_subset:
                fs(float(x[i]), float(y[i]), float(z[i]), int(d))

    def _vector_call(_n):
        for _, _, fv, d in coord_subset:
            fv(x[:_n], y[:_n], z[:_n], int(d))

    t_scalar, t_vec, sp = _bench_exact(_scalar_call, _vector_call, n_points)
    bench_rows.append({
        "Component": "Coordinate Transforms (subset)",
        "Scalar [s]": t_scalar,
        "Vectorized [s]": t_vec,
        "Speedup": sp,
    })

    # -- IGRF -----------------------------------------------------------
    igrf_scalar = igrf_vector = igrf_label = None
    if hasattr(geopack, "igrf_gsw") and hasattr(geopack, "igrf_gsw_vectorized"):
        igrf_scalar = geopack.igrf_gsw
        igrf_vector = geopack.igrf_gsw_vectorized
        igrf_label = "IGRF (GSW)"
    elif hasattr(geopack, "igrf_gsm") and hasattr(geopack, "igrf_gsm_vectorized"):
        igrf_scalar = geopack.igrf_gsm
        igrf_vector = geopack.igrf_gsm_vectorized
        igrf_label = "IGRF (GSM)"

    if igrf_scalar is not None:
        x, y, z = sample_xyz(n_points, (-10, 10), (-10, 10), (-10, 10), seed=SEED + 7)

        def _scalar_call(_n, _f=igrf_scalar, _x=x, _y=y, _z=z):
            for i in range(_n):
                _f(float(_x[i]), float(_y[i]), float(_z[i]))

        def _vector_call(_n, _f=igrf_vector, _x=x, _y=y, _z=z):
            _f(_x[:_n], _y[:_n], _z[:_n])

        t_scalar, t_vec, sp = _bench_exact(_scalar_call, _vector_call, n_points)
        bench_rows.append({
            "Component": igrf_label,
            "Scalar [s]": t_scalar,
            "Vectorized [s]": t_vec,
            "Speedup": sp,
        })
    else:
        bench_rows.append({
            "Component": "IGRF",
            "Scalar [s]": np.nan,
            "Vectorized [s]": np.nan,
            "Speedup": np.nan,
        })

    # -- External models ------------------------------------------------
    for model_name, f_s, f_v, params in model_functions:
        x, y, z = _make_xyz_for_models(n_points, model_name, seed=SEED)

        def _scalar_call(_n, _f_s=f_s, _params=params, _x=x, _y=y, _z=z):
            for i in range(_n):
                _f_s(_params, ps, float(_x[i]), float(_y[i]), float(_z[i]))

        def _vector_call(_n, _f_v=f_v, _params=params, _x=x, _y=y, _z=z):
            _f_v(_params, ps, _x[:_n], _y[:_n], _z[:_n])

        t_scalar, t_vec, sp = _bench_exact(_scalar_call, _vector_call, n_points)
        bench_rows.append({
            "Component": f"{model_name} Model",
            "Scalar [s]": t_scalar,
            "Vectorized [s]": t_vec,
            "Speedup": sp,
        })

    # -- Field line tracing (using vectorized field models) -------------
    trace_scalar = getattr(geopack, "trace", None)
    trace_vec = getattr(geopack, "trace_vectorized", None)

    TRACE_MAXLOOP = 200
    MAX_TRACE_SCALAR = 50
    TRACE_RETURN_FULL_PATH = True

    if trace_scalar is not None and trace_vec is not None:
        t_scalar, t_vec, sp, extrap, nsc = _bench_trace(
            n_target=n_points, max_scalar_lines=MAX_TRACE_SCALAR,
            n_repeats=N_REPEATS, maxloop=TRACE_MAXLOOP,
            return_full_path=TRACE_RETURN_FULL_PATH,
            strict_scalar_models=False,
        )
        label = "Field Line Tracing (vectorized field models)"
        if extrap:
            label += f" [scalar extrap from {nsc}]"
        bench_rows.append({
            "Component": label,
            "Scalar [s]": t_scalar,
            "Vectorized [s]": t_vec,
            "Speedup": sp,
        })
    else:
        bench_rows.append({
            "Component": "Field Line Tracing",
            "Scalar [s]": np.nan,
            "Vectorized [s]": np.nan,
            "Speedup": np.nan,
        })

    return pd.DataFrame(bench_rows)


def to_markdown(df, n_points):
    """Format a benchmark DataFrame as a README-ready Markdown table."""
    lines = [
        f"| Component | Scalar ({n_points} pts) [s] | Vectorized [s] | Speedup |",
        "|-----------|" + "-" * (len(f" Scalar ({n_points} pts) [s] ") - 2) + ":|---------------------:|--------:|",
    ]
    for _, row in df.iterrows():
        ts = row["Scalar [s]"]
        tv = row["Vectorized [s]"]
        sp = row["Speedup"]
        ts_s = "N/A" if (isinstance(ts, float) and np.isnan(ts)) else f"{ts:.3f}"
        tv_s = "N/A" if (isinstance(tv, float) and np.isnan(tv)) else f"{tv:.3f}"
        sp_s = "N/A" if (isinstance(sp, float) and np.isnan(sp)) else _fmt_speedup(sp)
        lines.append(f"| {row['Component']} | {ts_s} | {tv_s} | {sp_s} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="README-ready performance benchmarks")
    parser.add_argument("--plain", action="store_true",
                        help="Plain text output instead of Markdown")
    args = parser.parse_args()

    sizes = [100, 1000]

    for n in sizes:
        df = run_benchmarks(n)

        if args.plain:
            print(f"\n=== n = {n} points ===\n")
            print(df.to_string(index=False))
        else:
            print(to_markdown(df, n))
        print()


if __name__ == "__main__":
    main()
