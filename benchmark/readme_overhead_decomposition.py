"""
README-ready Cost Decomposition Benchmark

Separates the fixed per-call overhead of the vectorized functions from their
marginal per-point cost. The execution time of one vectorized call on an
n-element array is modeled as

    t_vec(n) ~= a + b * n

where `a` is a fixed overhead paid once per call and `b` is the marginal cost
per point. The scalar per-point cost is measured from loops over the same
points. Produces a compact Markdown table for the README with, per component:
overhead `a`, marginal cost `b`, scalar per-point cost, the per-point cost
ratio, and the break-even array size n* above which one vectorized call is
faster than the equivalent scalar loop.

Methodology notes:
  - Every timing cycles through several *distinct* chunks of randomly sampled
    points: timing a single fixed point repeatedly is unreliable because one
    point can hit an unusually cheap or expensive branch of a model.
  - Repetition counts are auto-calibrated per timing window; best of
    N_REPEATS is taken.
  - The slope `b` is fitted over n >= 128; the overhead `a` is estimated from
    the residuals at n <= 16 (t_vec(n) is not strictly linear for very small
    n, so single finite differences such as t(2) - t(1) are noisy).

Usage:
    python benchmark/readme_overhead_decomposition.py           # Markdown table
    python benchmark/readme_overhead_decomposition.py --plain   # plain text
"""

import argparse
import math
import time
from datetime import datetime

import numpy as np

from mageometry import geopack

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
N_REPEATS = 3
SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
POOL = max(SIZES)

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
ut = datetime(2020, 1, 1, 12, 0, 0).timestamp()
ps = geopack.recalc(ut)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now():
    return time.perf_counter()


def sample_xyz(n_points, xlim, ylim, zlim, seed=SEED):
    """Sample Cartesian points with float64 dtype."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(xlim[0], xlim[1], n_points)
    y = rng.uniform(ylim[0], ylim[1], n_points)
    z = rng.uniform(zlim[0], zlim[1], n_points)
    return x.astype(np.float64), y.astype(np.float64), z.astype(np.float64)


def _timed_over_chunks(call_one_chunk, chunks, n_repeats=N_REPEATS, min_window=0.02):
    """Best-of-repeats mean time of call_one_chunk(chunk), cycling over
    several distinct data chunks."""
    call_one_chunk(chunks[0])  # warmup
    t0 = _now()
    for c in chunks:
        call_one_chunk(c)
    t_pass = _now() - t0
    n_pass = max(1, int(np.ceil(min_window / max(t_pass, 1e-9))))
    best = np.inf
    for _ in range(n_repeats):
        t0 = _now()
        for _ in range(n_pass):
            for c in chunks:
                call_one_chunk(c)
        best = min(best, (_now() - t0) / (n_pass * len(chunks)))
    return float(best)


def measure_small_n(scalar_call, vector_call, xlim, ylim, zlim):
    """Time scalar loops and single vectorized calls over SIZES."""
    x, y, z = sample_xyz(POOL, xlim, ylim, zlim)
    rows = []
    for n in SIZES:
        n_chunk_s = max(1, min(8, 512 // n, POOL // n))
        n_chunk_v = max(1, min(8, POOL // n))
        chunks_s = [(x[i*n:(i+1)*n], y[i*n:(i+1)*n], z[i*n:(i+1)*n]) for i in range(n_chunk_s)]
        chunks_v = [(x[i*n:(i+1)*n], y[i*n:(i+1)*n], z[i*n:(i+1)*n]) for i in range(n_chunk_v)]

        def scalar_loop(chunk):
            cx, cy, cz = chunk
            for i in range(len(cx)):
                scalar_call(float(cx[i]), float(cy[i]), float(cz[i]))

        def vector_once(chunk):
            vector_call(chunk[0], chunk[1], chunk[2])

        rows.append({
            "n": n,
            "scalar_s": _timed_over_chunks(scalar_loop, chunks_s),
            "vector_s": _timed_over_chunks(vector_once, chunks_v),
        })
    return rows


def decompose_costs(rows):
    """Fit t(n) ~= a + b*n: slope from n >= 128, overhead from n <= 16 residuals."""
    n = np.array([r["n"] for r in rows], dtype=float)
    ts = np.array([r["scalar_s"] for r in rows])
    tv = np.array([r["vector_s"] for r in rows])
    big = n >= 128
    b_s = np.polyfit(n[big], ts[big], 1)[0]
    b_v = np.polyfit(n[big], tv[big], 1)[0]
    a_v = float(np.median((tv - b_v * n)[n <= 16]))
    ratio = ts / tv
    if np.all(ratio >= 1.0):
        n_cross = 1.0
    elif np.all(ratio < 1.0):
        n_cross = float("inf")
    else:  # log-interpolate the first upward crossing of speedup = 1
        hi = int(np.argmax(ratio >= 1.0))
        lo = hi - 1
        f = -np.log(ratio[lo]) / (np.log(ratio[hi]) - np.log(ratio[lo]))
        n_cross = float(np.exp(np.log(n[lo]) + f * (np.log(n[hi]) - np.log(n[lo]))))
    return {"b_scalar": float(b_s), "b_vector": float(b_v), "a_vector": a_v,
            "slope_ratio": float(b_s / b_v), "n_cross": n_cross}


# ---------------------------------------------------------------------------
# Model parameters (same as readme_benchmarks.py)
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


def _target(name, f_s, f_v, params, xlim, ylim=(-15, 15), zlim=(-15, 15)):
    if params is None:  # IGRF-style signature (x, y, z)
        return (name, f_s, f_v, xlim, ylim, zlim)
    return (name,
            lambda x, y, z: f_s(params, ps, x, y, z),
            lambda x, y, z: f_v(params, ps, x, y, z),
            xlim, ylim, zlim)


decomp_targets = [
    _target("T89", geopack.t89, geopack.t89_vectorized, 4, (-25.0, 12.0)),
    _target("T96", geopack.t96, geopack.t96_vectorized, parmod_default, (-25.0, 12.0)),
    _target("T01", geopack.t01, geopack.t01_vectorized, parmod_t01, (-15.0, 12.0)),
    _target("T04", geopack.t04, geopack.t04_vectorized, parmod_t04, (-15.0, 12.0)),
]
if hasattr(geopack, "igrf_gsw") and hasattr(geopack, "igrf_gsw_vectorized"):
    decomp_targets.append(_target("IGRF (GSW)", geopack.igrf_gsw,
                                  geopack.igrf_gsw_vectorized, None,
                                  (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)))


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def _sig2(x):
    """Two significant figures, no exponent notation (for ms-scale values)."""
    if x == 0:
        return "0"
    d = -int(math.floor(math.log10(abs(x)))) + 1
    return f"{round(x, d):.{max(d, 0)}f}"


def _fmt_row(name, d):
    a_ms = _sig2(d["a_vector"] * 1e3)
    b_us = f"{d['b_vector']*1e6:.2f}" if d["b_vector"] * 1e6 < 1 else f"{d['b_vector']*1e6:.1f}"
    s_us = f"{d['b_scalar']*1e6:.1f}" if d["b_scalar"] * 1e6 < 100 else f"{d['b_scalar']*1e6:.0f}"
    ratio = f"**{d['slope_ratio']:.0f}x**"
    ncr = "N/A" if not np.isfinite(d["n_cross"]) else str(int(math.floor(d["n_cross"] + 0.5)))
    return name, a_ms, b_us, s_us, ratio, ncr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="README-ready cost decomposition benchmark")
    parser.add_argument("--plain", action="store_true",
                        help="Plain text output instead of Markdown")
    args = parser.parse_args()

    results = {}
    for name, s_call, v_call, xlim, ylim, zlim in decomp_targets:
        rows = measure_small_n(s_call, v_call, xlim, ylim, zlim)
        results[name] = decompose_costs(rows)

    if args.plain:
        print(f"{'Component':12s} {'a [ms/call]':>12s} {'b [us/pt]':>10s} "
              f"{'scalar [us/pt]':>15s} {'ratio':>8s} {'n*':>5s}")
        for name, d in results.items():
            r = _fmt_row(name, d)
            print(f"{r[0]:12s} {r[1]:>12s} {r[2]:>10s} {r[3]:>15s} "
                  f"{r[4].strip('*'):>8s} {r[5]:>5s}")
    else:
        print("| Component | Overhead a [ms/call] | Marginal b [µs/point] "
              "| Scalar [µs/point] | Per-point ratio | Break-even n* |")
        print("|-----------|---------------------:|----------------------:"
              "|------------------:|----------------:|--------------:|")
        for name, d in results.items():
            print("| {} | {} | {} | {} | {} | {} |".format(*_fmt_row(name, d)))


if __name__ == "__main__":
    main()
