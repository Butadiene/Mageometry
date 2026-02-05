#!/usr/bin/env python3
"""
validate_tracing_accuracy.py

スカラー版(RK4)と、ベクトル版(trace_vectorized / trace_vectorized_no_interp)を
「既知の磁場=双極子(dipole)」でトレースして精度検証するテストコードです。

ポイント
- 双極子の磁力線は解析的に r = L sin^2(theta)（theta: colatitude）を満たすので、
  開始点(L-shell)から Earth's surface (r=1) のフットポイントが解析的に求められます。
- ベクトル版は外部磁場(exname)が必須ですが、検証では外部磁場をゼロにパッチして
  「内部=dipoleのみ」にしています（アルゴリズム精度検証目的）。

実行例
    python validate_tracing_accuracy.py

（パッケージとして配置している場合）
    python -m geopack.validate_tracing_accuracy

依存
    numpy, matplotlib
    geopack（あなたのプロジェクトの geopack）
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (unused but kept for consistency)
import pandas as pd  # noqa: F401
import time  # noqa: F401
from scipy import stats  # noqa: F401

import geopack
import geopack.trace_field_lines_vectorized as tfv
import geopack.trace_field_lines_vectorized_nointerp as tfvni
from geopack.trace_field_lines_vectorized import trace_vectorized
from geopack.trace_field_lines_vectorized_nointerp import trace_vectorized_no_interp

# Set up plotting (user-style)
plt.style.use("default")
plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["font.size"] = 12

# --------------------------
# 2) Scalar tracer (RK4) - based on your scalar example
# --------------------------
def trace_field_line_rk4(
    model_func: Callable[[np.ndarray, float, float, float, float], Tuple[float, float, float]],
    params: np.ndarray,
    ps: float,
    start_pos: Tuple[float, float, float],
    step_size: float = 0.05,
    max_steps: int = 20000,
    r_stop_inner: float = 1.0,
    r_stop_outer: float = 50.0,
) -> np.ndarray:
    """
    Scalar field line tracer using Runge-Kutta 4th order method (固定ステップ).
    元のスカラー例と同じく B を正規化した方向ベクトルに沿って積分します。

    NOTE:
      r <= r_stop_inner を検知して止めるだけだと「境界ちょうど」にはならないので、
      フットポイント評価では後段で線形補間します。
    """
    positions: List[Tuple[float, float, float]] = [start_pos]
    x, y, z = start_pos

    for _ in range(max_steps):
        bx, by, bz = model_func(params, ps, x, y, z)
        b_mag = math.sqrt(bx * bx + by * by + bz * bz)
        if b_mag < 1e-12:
            break

        # unit direction
        dx, dy, dz = bx / b_mag, by / b_mag, bz / b_mag

        # RK4
        k1x, k1y, k1z = step_size * dx, step_size * dy, step_size * dz

        bx2, by2, bz2 = model_func(params, ps, x + 0.5 * k1x, y + 0.5 * k1y, z + 0.5 * k1z)
        b_mag2 = math.sqrt(bx2 * bx2 + by2 * by2 + bz2 * bz2)
        if b_mag2 < 1e-12:
            break
        k2x, k2y, k2z = step_size * bx2 / b_mag2, step_size * by2 / b_mag2, step_size * bz2 / b_mag2

        bx3, by3, bz3 = model_func(params, ps, x + 0.5 * k2x, y + 0.5 * k2y, z + 0.5 * k2z)
        b_mag3 = math.sqrt(bx3 * bx3 + by3 * by3 + bz3 * bz3)
        if b_mag3 < 1e-12:
            break
        k3x, k3y, k3z = step_size * bx3 / b_mag3, step_size * by3 / b_mag3, step_size * bz3 / b_mag3

        bx4, by4, bz4 = model_func(params, ps, x + k3x, y + k3y, z + k3z)
        b_mag4 = math.sqrt(bx4 * bx4 + by4 * by4 + bz4 * bz4)
        if b_mag4 < 1e-12:
            break
        k4x, k4y, k4z = step_size * bx4 / b_mag4, step_size * by4 / b_mag4, step_size * bz4 / b_mag4

        x += (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
        y += (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
        z += (k1z + 2.0 * k2z + 2.0 * k3z + k4z) / 6.0
        positions.append((x, y, z))

        r = math.sqrt(x * x + y * y + z * z)
        if r <= r_stop_inner or r >= r_stop_outer:
            break

    return np.array(positions, dtype=float)


# --------------------------
# 3) Geometry utilities + analytic dipole footprint
# --------------------------

def analytic_dipole_B(x, y, z):
    """
    解析的な「軸対称・tilt=0」の双極子磁場ベクトル（スケール定数なし）。
    m = (0,0,1) として
        B = (3 r (m·r)/r^5 - m/r^3)
    を用いる。

    返り値は (Bx, By, Bz)。x,y,z はスカラー/配列どちらでもOK。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    r2 = x * x + y * y + z * z
    r = np.sqrt(r2)
    r5 = r2 * r2 * r

    # avoid divide-by-zero
    eps = 1e-30
    r5 = np.where(r5 < eps, eps, r5)

    bx = 3.0 * x * z / r5
    by = 3.0 * y * z / r5
    bz = (3.0 * z * z - r2) / r5
    return bx, by, bz

def cart_to_sph(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Returns (r, theta, phi)
    theta: colatitude [0..pi], phi: longitude [-pi..pi]
    """
    r = math.sqrt(x * x + y * y + z * z)
    theta = math.acos(z / r) if r > 0 else 0.0
    phi = math.atan2(y, x)
    return r, theta, phi


def sph_to_cart(r: float, theta: float, phi: float) -> Tuple[float, float, float]:
    st = math.sin(theta)
    return (r * st * math.cos(phi), r * st * math.sin(phi), r * math.cos(theta))


def dipole_L_shell_from_point(x: float, y: float, z: float) -> float:
    """
    Dipole field line invariant:
      r = L sin^2(theta)  =>  L = r / sin^2(theta)
    """
    r, theta, _ = cart_to_sph(x, y, z)
    s = math.sin(theta)
    if s == 0:
        return float("inf")
    return r / (s * s)


def analytic_dipole_footpoints_from_equator(L: float, phi: float, r0: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Start at equator (theta=pi/2) with radius r=L.
    Footpoints at r=r0 satisfy:
      r0 = L sin^2(theta_fp)  =>  sin(theta_fp) = sqrt(r0/L)
    theta_fp in (0, pi/2]  for north, (pi/2, pi) for south.
    """
    sin_th = math.sqrt(r0 / L)
    # Clamp for safety
    sin_th = min(1.0, max(0.0, sin_th))
    th_n = math.asin(sin_th)         # north colatitude
    th_s = math.pi - th_n            # south colatitude
    xn, yn, zn = sph_to_cart(r0, th_n, phi)
    xs, ys, zs = sph_to_cart(r0, th_s, phi)
    return np.array([xn, yn, zn]), np.array([xs, ys, zs])


def interpolate_to_radius(p0: np.ndarray, p1: np.ndarray, r_target: float = 1.0) -> np.ndarray:
    """
    線分 p0->p1 が r_target をまたぐとして、|p|=r_target になる点を線形補間で返す。
    """
    r0 = np.linalg.norm(p0)
    r1 = np.linalg.norm(p1)
    if abs(r1 - r0) < 1e-15:
        return p1.copy()
    t = (r_target - r0) / (r1 - r0)
    t = float(np.clip(t, 0.0, 1.0))
    return p0 + t * (p1 - p0)


def footprint_from_path(path: np.ndarray, r0: float = 1.0) -> Optional[np.ndarray]:
    """
    パスの中から r0 を跨いだ最初の区間を見つけて、線形補間で r=r0 の点を返す。
    """
    if len(path) < 2:
        return None
    r = np.linalg.norm(path, axis=1)
    # find first crossing from >r0 to <=r0
    for i in range(1, len(path)):
        if (r[i - 1] > r0) and (r[i] <= r0):
            return interpolate_to_radius(path[i - 1], path[i], r0)
    # already inside?
    if r[-1] <= r0:
        return path[-1].copy()
    return None


def lat_lon_deg(p: np.ndarray) -> Tuple[float, float]:
    """
    Geographic-like lat/lon (but here it's dipole coords)
    lat = asin(z/r), lon = atan2(y,x)
    """
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    r = math.sqrt(x * x + y * y + z * z)
    if r == 0:
        return 0.0, 0.0
    lat = math.degrees(math.asin(z / r))
    lon = math.degrees(math.atan2(y, x))
    return lat, lon


def wrap_lon_err_deg(lon: float, lon_ref: float) -> float:
    """
    lon-lon_ref を [-180,180] に折り返す
    """
    d = lon - lon_ref
    while d > 180.0:
        d -= 360.0
    while d < -180.0:
        d += 360.0
    return d


# --------------------------
# 4) Validation core
# --------------------------
@dataclass
class CaseResult:
    L: float
    phi_deg: float
    hemi: str  # "N" or "S"

    # analytic
    expected_xyz: np.ndarray
    expected_lat: float
    expected_lon: float

    # scalar
    scalar_xyz: Optional[np.ndarray]
    scalar_lat_err: Optional[float]
    scalar_lon_err: Optional[float]
    scalar_dist_err: Optional[float]

    # vectorized (interp)
    vec_xyz: Optional[np.ndarray]
    vec_lat_err: Optional[float]
    vec_lon_err: Optional[float]
    vec_dist_err: Optional[float]
    vec_status: Optional[int]

    # vectorized (nointerp outer)
    vec_ni_xyz: Optional[np.ndarray]
    vec_ni_lat_err: Optional[float]
    vec_ni_lon_err: Optional[float]
    vec_ni_dist_err: Optional[float]
    vec_ni_status: Optional[int]


def patch_external_to_zero(tfv_module) -> None:
    """
    ベクトル版は外部磁場モデルが必須なので、検証用に外部磁場をゼロに置き換える。
    """
    def zero_external(exname, parmod, ps, x, y, z):
        x = np.asarray(x)
        return np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)

    tfv_module.call_external_model_vectorized = zero_external  # type: ignore


def run_dipole_validation(
    L_values: List[float],
    phi_deg_values: List[float],
    step_scalar: float = 0.02,
    max_steps_scalar: int = 30000,
    r0: float = 1.0,
    rlim: float = 50.0,
    maxloop_vec: int = 5000,
    make_plots: bool = True,
) -> List[CaseResult]:
    # (imports moved to top of file)

    # 統一：双極子 tilt を 0 にして、解析解と揃える
    # （geopack.dip が global tilt を参照する実装の場合を想定）
    # geopack.trace_field_lines_vectorized の実装では
    #   from . import geopack as gp
    #   psi = gp.psi
    # のように geopack/geopack.py (module名: geopack.geopack) を参照します。
    # そのため gp.psi が存在しない環境では AttributeError になります。
    # ここで明示的に gp.psi を作成・設定しておきます。
    try:
        from geopack import geopack as gp  # type: ignore
        # 解析解(dipole)と一致させるため tilt=0
        gp.psi = 0.0  # type: ignore[attr-defined]
        # geopack.psi も参照される可能性があるので同期
        geopack.psi = gp.psi  # type: ignore[attr-defined]
    except Exception:
        # 最低限 geopack.psi があれば vector 側の gp.psi 参照以外では動くので設定だけ試す
        try:
            geopack.psi = 0.0  # type: ignore[attr-defined]
        except Exception:
            pass

    patch_external_to_zero(tfv)
    patch_external_to_zero(tfvni)

    # ベクトル版の内部 dipole を geopack 実装に依存しない解析式へ差し替え
    # （geopack.geopack.dip は IGRF 係数 g/h に依存し、環境によって未初期化で落ちるため）
    tfv.dip = analytic_dipole_B  # type: ignore[attr-defined]
    tfvni.dip = analytic_dipole_B  # type: ignore[attr-defined]


    # scalar: analytic dipole (tilt=0)
    def dipole_model(_params, _ps, x, y, z):
        return analytic_dipole_B(x, y, z)

    params_dummy = np.zeros(10, dtype=float)
    ps_dummy = 0.0

    # === prepare starting points ===
    starts = []
    meta = []
    for L in L_values:
        for phi_deg in phi_deg_values:
            phi = math.radians(phi_deg)
            x0 = L * math.cos(phi)
            y0 = L * math.sin(phi)
            z0 = 0.0
            starts.append((x0, y0, z0))
            meta.append((L, phi_deg, phi))

    # === vectorized endpoints (two dirs) ===
    xi = np.array([s[0] for s in starts], dtype=float)
    yi = np.array([s[1] for s in starts], dtype=float)
    zi = np.array([s[2] for s in starts], dtype=float)

    # run both directions
    xf_p, yf_p, zf_p, status_p = trace_vectorized(  # type: ignore
        xi, yi, zi, dir=+1, rlim=rlim, r0=r0, parmod=2, exname="t89", inname="dipole", maxloop=maxloop_vec
    )
    xf_m, yf_m, zf_m, status_m = trace_vectorized(  # type: ignore
        xi, yi, zi, dir=-1, rlim=rlim, r0=r0, parmod=2, exname="t89", inname="dipole", maxloop=maxloop_vec
    )

    xfni_p, yfni_p, zfni_p, statusni_p = trace_vectorized_no_interp(  # type: ignore
        xi, yi, zi, dir=+1, rlim=rlim, r0=r0, parmod=2, exname="t89", inname="dipole", maxloop=maxloop_vec
    )
    xfni_m, yfni_m, zfni_m, statusni_m = trace_vectorized_no_interp(  # type: ignore
        xi, yi, zi, dir=-1, rlim=rlim, r0=r0, parmod=2, exname="t89", inname="dipole", maxloop=maxloop_vec
    )

    # === scalar paths (two directions) ===
    scalar_fp_plus = []  # step positive
    scalar_fp_minus = []  # step negative
    for s in starts:
        path_p = trace_field_line_rk4(dipole_model, params_dummy, ps_dummy, s, step_size=+abs(step_scalar), max_steps=max_steps_scalar)
        path_m = trace_field_line_rk4(dipole_model, params_dummy, ps_dummy, s, step_size=-abs(step_scalar), max_steps=max_steps_scalar)
        fp_p = footprint_from_path(path_p, r0=r0)
        fp_m = footprint_from_path(path_m, r0=r0)
        scalar_fp_plus.append(fp_p)
        scalar_fp_minus.append(fp_m)

    # === assemble results ===
    results: List[CaseResult] = []

    for idx, (L, phi_deg, phi) in enumerate(meta):
        expN, expS = analytic_dipole_footpoints_from_equator(L, phi, r0=r0)

        # expected lat/lon
        expN_lat, expN_lon = lat_lon_deg(expN)
        expS_lat, expS_lon = lat_lon_deg(expS)

        # Vectorized: choose which endpoint is N/S by sign of z
        v_p = np.array([xf_p[idx], yf_p[idx], zf_p[idx]], dtype=float)
        v_m = np.array([xf_m[idx], yf_m[idx], zf_m[idx]], dtype=float)
        vn_p = np.array([xfni_p[idx], yfni_p[idx], zfni_p[idx]], dtype=float)
        vn_m = np.array([xfni_m[idx], yfni_m[idx], zfni_m[idx]], dtype=float)

        v_p_is_n = v_p[2] >= 0
        vN = v_p if v_p_is_n else v_m
        vS = v_m if v_p_is_n else v_p
        vN_stat = int(status_p[idx]) if v_p_is_n else int(status_m[idx])
        vS_stat = int(status_m[idx]) if v_p_is_n else int(status_p[idx])

        vn_p_is_n = vn_p[2] >= 0
        vnN = vn_p if vn_p_is_n else vn_m
        vnS = vn_m if vn_p_is_n else vn_p
        vnN_stat = int(statusni_p[idx]) if vn_p_is_n else int(statusni_m[idx])
        vnS_stat = int(statusni_m[idx]) if vn_p_is_n else int(statusni_p[idx])

        # Scalar: which step gave north?
        s_p = scalar_fp_plus[idx]
        s_m = scalar_fp_minus[idx]
        # fallback if None
        sN, sS = None, None
        if s_p is not None and s_m is not None:
            s_p_is_n = s_p[2] >= 0
            sN = s_p if s_p_is_n else s_m
            sS = s_m if s_p_is_n else s_p
        elif s_p is not None:
            sN = s_p if s_p[2] >= 0 else None
            sS = s_p if s_p[2] < 0 else None
        elif s_m is not None:
            sN = s_m if s_m[2] >= 0 else None
            sS = s_m if s_m[2] < 0 else None

        # helper to compute errors
        def _errs(p: Optional[np.ndarray], pref: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
            if p is None:
                return None, None, None
            lat, lon = lat_lon_deg(p)
            lat_ref, lon_ref = lat_lon_deg(pref)
            return (lat - lat_ref, wrap_lon_err_deg(lon, lon_ref), float(np.linalg.norm(p - pref)))

        # North
        s_lat_e, s_lon_e, s_dist_e = _errs(sN, expN)
        v_lat_e, v_lon_e, v_dist_e = _errs(vN, expN)
        vn_lat_e, vn_lon_e, vn_dist_e = _errs(vnN, expN)
        results.append(
            CaseResult(
                L=L,
                phi_deg=phi_deg,
                hemi="N",
                expected_xyz=expN,
                expected_lat=expN_lat,
                expected_lon=expN_lon,
                scalar_xyz=sN,
                scalar_lat_err=s_lat_e,
                scalar_lon_err=s_lon_e,
                scalar_dist_err=s_dist_e,
                vec_xyz=vN,
                vec_lat_err=v_lat_e,
                vec_lon_err=v_lon_e,
                vec_dist_err=v_dist_e,
                vec_status=vN_stat,
                vec_ni_xyz=vnN,
                vec_ni_lat_err=vn_lat_e,
                vec_ni_lon_err=vn_lon_e,
                vec_ni_dist_err=vn_dist_e,
                vec_ni_status=vnN_stat,
            )
        )

        # South
        s_lat_e, s_lon_e, s_dist_e = _errs(sS, expS)
        v_lat_e, v_lon_e, v_dist_e = _errs(vS, expS)
        vn_lat_e, vn_lon_e, vn_dist_e = _errs(vnS, expS)
        results.append(
            CaseResult(
                L=L,
                phi_deg=phi_deg,
                hemi="S",
                expected_xyz=expS,
                expected_lat=expS_lat,
                expected_lon=expS_lon,
                scalar_xyz=sS,
                scalar_lat_err=s_lat_e,
                scalar_lon_err=s_lon_e,
                scalar_dist_err=s_dist_e,
                vec_xyz=vS,
                vec_lat_err=v_lat_e,
                vec_lon_err=v_lon_e,
                vec_dist_err=v_dist_e,
                vec_status=vS_stat,
                vec_ni_xyz=vnS,
                vec_ni_lat_err=vn_lat_e,
                vec_ni_lon_err=vn_lon_e,
                vec_ni_dist_err=vn_dist_e,
                vec_ni_status=vnS_stat,
            )
        )

    # === report ===
    print("\n=== Dipole footprint validation (r=1) ===")
    print("Columns: L, phi_deg, hemi, |err| [Re], lat_err [deg], lon_err [deg], status")
    print("- scalar: RK4 fixed step")
    print("- vec   : trace_vectorized (interp)")
    print("- vecNI : trace_vectorized_no_interp (outer no-interp)")
    print()

    def fmt(x):
        if x is None:
            return "None"
        return f"{x:+.3e}"

    for r in results:
        print(
            f"L={r.L:>4.1f} phi={r.phi_deg:>5.1f} {r.hemi} | "
            f"scalar d={fmt(r.scalar_dist_err)} lat={fmt(r.scalar_lat_err)} lon={fmt(r.scalar_lon_err)} | "
            f"vec d={fmt(r.vec_dist_err)} lat={fmt(r.vec_lat_err)} lon={fmt(r.vec_lon_err)} st={r.vec_status} | "
            f"vecNI d={fmt(r.vec_ni_dist_err)} lat={fmt(r.vec_ni_lat_err)} lon={fmt(r.vec_ni_lon_err)} st={r.vec_ni_status}"
        )

    # === plots ===
    if make_plots and plt is not None:
        # Error vs L (distance)
        Ls = np.array([r.L for r in results if r.hemi == "N"], dtype=float)
        # pick north hemisphere errors to avoid duplicating L
        scalar_err = np.array([r.scalar_dist_err for r in results if r.hemi == "N"], dtype=float)
        vec_err = np.array([r.vec_dist_err for r in results if r.hemi == "N"], dtype=float)
        vecni_err = np.array([r.vec_ni_dist_err for r in results if r.hemi == "N"], dtype=float)

        plt.figure(figsize=(8, 5))
        plt.plot(Ls, scalar_err, marker="o", label="scalar RK4")
        plt.plot(Ls, vec_err, marker="o", label="vectorized (interp)")
        plt.plot(Ls, vecni_err, marker="o", label="vectorized (nointerp outer)")
        plt.yscale("log")
        plt.xlabel("L-shell (start radius at equator)")
        plt.ylabel("Footpoint distance error |p - p_analytic|  [Re]")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("dipole_footpoint_error.png", dpi=160)
        print("\nSaved: dipole_footpoint_error.png")

        # Example field line shape in XZ plane for phi=0, L=6
        # (scalar: full path, vector: request full path for single trace)
        try:
            L_example = float(L_values[len(L_values)//2])
            phi_example = 0.0
            start = (L_example, 0.0, 0.0)
            # analytic curve in XZ plane at phi=0: x = r sin(theta), z = r cos(theta), r = L sin^2(theta)
            th = np.linspace(0.01, math.pi - 0.01, 800)
            r_an = L_example * (np.sin(th) ** 2)
            x_an = r_an * np.sin(th)
            z_an = r_an * np.cos(th)

            # scalar path (both directions)
            path_p = trace_field_line_rk4(dipole_model, params_dummy, ps_dummy, start, step_size=+abs(step_scalar), max_steps=max_steps_scalar)
            path_m = trace_field_line_rk4(dipole_model, params_dummy, ps_dummy, start, step_size=-abs(step_scalar), max_steps=max_steps_scalar)
            path_s = np.vstack([path_m[::-1], path_p[1:]])

            # vector full path (two dirs)
            xf, yf, zf, xx, yy, zz, st = trace_vectorized(  # type: ignore
                start[0], start[1], start[2], dir=+1, rlim=rlim, r0=r0, parmod=2, exname="t89", inname="dipole",
                maxloop=maxloop_vec, return_full_path=True
            )
            xf2, yf2, zf2, xx2, yy2, zz2, st2 = trace_vectorized(  # type: ignore
                start[0], start[1], start[2], dir=-1, rlim=rlim, r0=r0, parmod=2, exname="t89", inname="dipole",
                maxloop=maxloop_vec, return_full_path=True
            )
            path_v = np.vstack([np.column_stack([xx2, yy2, zz2])[::-1], np.column_stack([xx, yy, zz])[1:]])

            plt.figure(figsize=(7, 7))
            plt.plot(x_an, z_an, label="analytic dipole", linewidth=2)
            plt.plot(path_s[:, 0], path_s[:, 2], label="scalar RK4", linewidth=2)
            plt.plot(path_v[:, 0], path_v[:, 2], label="vectorized RK5", linewidth=2)
            # Earth circle
            earth = plt.Circle((0, 0), 1.0, alpha=0.2)
            plt.gca().add_patch(earth)
            plt.gca().set_aspect("equal", adjustable="box")
            plt.xlabel("X [Re]")
            plt.ylabel("Z [Re]")
            plt.title(f"Dipole field line example (L={L_example:.1f}, phi={phi_example:.1f}°)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig("dipole_field_line_example.png", dpi=160)
            print("Saved: dipole_field_line_example.png")
        except Exception as e:
            print(f"(plot example skipped due to error: {e})")

    return results


def main():
    # 検証ケース：磁気赤道上の開始点 r=L, z=0
    L_values = [2, 3, 4, 5, 6, 7, 8]
    phi_deg_values = [0.0, 30.0, 60.0]  # 経度依存は無いはず（dipoleは軸対称）

    # scalarのステップを変えて誤差を見るのが有効です
    results = run_dipole_validation(
        L_values=L_values,
        phi_deg_values=phi_deg_values,
        step_scalar=0.02,
        max_steps_scalar=40000,
        r0=1.0,
        rlim=50.0,
        maxloop_vec=6000,
        make_plots=True,
    )

    # まとめ（最大誤差）
    scalar_dist = [r.scalar_dist_err for r in results if r.scalar_dist_err is not None]
    vec_dist = [r.vec_dist_err for r in results if r.vec_dist_err is not None]
    vecni_dist = [r.vec_ni_dist_err for r in results if r.vec_ni_dist_err is not None]

    print("\n=== Summary (max |distance error|) ===")
    if scalar_dist:
        print(f"scalar RK4: {max(scalar_dist):.3e} Re")
    if vec_dist:
        print(f"vectorized (interp): {max(vec_dist):.3e} Re")
    if vecni_dist:
        print(f"vectorized (nointerp outer): {max(vecni_dist):.3e} Re")


if __name__ == "__main__":
    main()