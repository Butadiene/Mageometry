"""
Trace src_prc_vectorized to find where -32 nT comes from.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geopack import t01
from geopack.ring_current_vectorized import src_prc_vectorized, prc_symm_vectorized, prc_quad_vectorized


def trace_src_prc():
    """Trace src_prc_vectorized step by step."""
    print("SRC_PRC_VECTORIZED TRACE")
    print("=" * 80)
    
    # Parameters
    ps = -0.1
    x = -12.899830  # Already scaled by xappa
    y = 0.0
    z = 0.0
    sc_sy = 1.312290
    sc_pr = 0.822513
    phi = 1.570796
    
    print(f"Input parameters:")
    print(f"  x={x:.6f}, y={y:.6f}, z={z:.6f}")
    print(f"  sc_pr={sc_pr:.6f}, phi={phi:.6f}")
    
    # Arrays
    x_arr = np.array([x])
    y_arr = np.array([y])
    z_arr = np.array([z])
    
    # Manually trace what src_prc_vectorized does
    print("\n" + "=" * 80)
    print("MANUAL TRACE OF src_prc_vectorized:")
    
    # 1. Coordinate transformation
    sps = np.sin(ps)
    cps = np.cos(ps)
    
    xt = x_arr * cps - z_arr * sps
    zt = x_arr * sps + z_arr * cps
    
    print(f"\n1. Tilt rotation:")
    print(f"   xt = x*cos(ps) - z*sin(ps) = {x}*{cps:.6f} - {z}*{sps:.6f} = {xt[0]:.6f}")
    print(f"   zt = x*sin(ps) + z*cos(ps) = {x}*{sps:.6f} + {z}*{cps:.6f} = {zt[0]:.6f}")
    
    # 2. Scale for PRC
    xta = xt / sc_pr
    yta = y_arr / sc_pr
    zta = zt / sc_pr
    
    print(f"\n2. Scale for PRC:")
    print(f"   xta = xt/sc_pr = {xt[0]:.6f}/{sc_pr:.6f} = {xta[0]:.6f}")
    print(f"   yta = y/sc_pr = {y}/{sc_pr:.6f} = {yta[0]:.6f}")
    print(f"   zta = zt/sc_pr = {zt[0]:.6f}/{sc_pr:.6f} = {zta[0]:.6f}")
    
    # 3. PRC symmetric part
    bxa_s, bya_s, bza_s = prc_symm_vectorized(xta, yta, zta)
    print(f"\n3. PRC symmetric:")
    print(f"   bxa_s={bxa_s[0]:.6f}, bya_s={bya_s[0]:.6f}, bza_s={bza_s[0]:.6f}")
    
    # 4. Rotate for quadrupole
    cp = np.cos(phi)
    sp = np.sin(phi)
    xr = xta * cp - yta * sp
    yr = xta * sp + yta * cp
    
    print(f"\n4. Rotate for quadrupole (phi={phi:.6f}):")
    print(f"   xr = xta*cos(phi) - yta*sin(phi) = {xta[0]:.6f}*{cp:.6f} - {yta[0]:.6f}*{sp:.6f} = {xr[0]:.6f}")
    print(f"   yr = xta*sin(phi) + yta*cos(phi) = {xta[0]:.6f}*{sp:.6f} + {yta[0]:.6f}*{cp:.6f} = {yr[0]:.6f}")
    
    # 5. PRC quadrupole
    bxa_qr, bya_qr, bza_q = prc_quad_vectorized(xr, yr, zta)
    print(f"\n5. PRC quadrupole (in rotated coords):")
    print(f"   bxa_qr={bxa_qr[0]:.6f}, bya_qr={bya_qr[0]:.6f}, bza_q={bza_q[0]:.6f}")
    
    # 6. Transform quadrupole back
    bxa_q = bxa_qr * cp + bya_qr * sp
    bya_q = -bxa_qr * sp + bya_qr * cp
    
    print(f"\n6. Transform quadrupole back:")
    print(f"   bxa_q = bxa_qr*cos(phi) + bya_qr*sin(phi) = {bxa_q[0]:.6f}")
    print(f"   bya_q = -bxa_qr*sin(phi) + bya_qr*cos(phi) = {bya_q[0]:.6f}")
    
    # 7. Total PRC field
    bxp = bxa_s + bxa_q
    byp = bya_s + bya_q
    bzp = bza_s + bza_q
    
    print(f"\n7. Total PRC field:")
    print(f"   bxp = {bxp[0]:.6f}")
    print(f"   byp = {byp[0]:.6f}")
    print(f"   bzp = {bzp[0]:.6f}")
    
    # 8. Transform back to GSM
    bxprc = bxp * cps + bzp * sps
    byprc = byp
    bzprc = bzp * cps - bxp * sps
    
    print(f"\n8. Transform back to GSM:")
    print(f"   bxprc = bxp*cos(ps) + bzp*sin(ps) = {bxp[0]:.6f}*{cps:.6f} + {bzp[0]:.6f}*{sps:.6f} = {bxprc[0]:.6f}")
    print(f"   byprc = byp = {byprc[0]:.6f}")
    print(f"   bzprc = bzp*cos(ps) - bxp*sin(ps) = {bzp[0]:.6f}*{cps:.6f} - {bxp[0]:.6f}*{sps:.6f} = {bzprc[0]:.6f}")
    
    print(f"\nFINAL PRC: Bz = {bzprc[0]:.6f}")
    
    # Compare with actual src_prc_vectorized
    print("\n" + "=" * 80)
    print("ACTUAL src_prc_vectorized CALL:")
    
    bxsrc, bysrc, bzsrc, bxprc_actual, byprc_actual, bzprc_actual = src_prc_vectorized(
        2, sc_sy, sc_pr, phi, ps, x_arr, y_arr, z_arr
    )
    
    print(f"PRC from src_prc_vectorized: Bz = {bzprc_actual[0]:.6f}")
    print(f"Difference from manual trace: {bzprc_actual[0] - bzprc[0]:.6f}")
    
    # The -32 nT must come from full_rc_vectorized
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print(f"src_prc_vectorized gives PRC Bz = {bzprc_actual[0]:.6f}")
    print(f"but full_rc_vectorized gives PRC Bz = -32.3")
    print(f"The error must be in full_rc_vectorized, not src_prc_vectorized!")


if __name__ == "__main__":
    trace_src_prc()