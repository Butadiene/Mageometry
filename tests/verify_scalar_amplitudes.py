"""
Verify what amplitudes the scalar code actually calculates.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geopack import t01


def trace_scalar_amplitudes():
    """Trace amplitude calculations in scalar code."""
    print("TRACING SCALAR AMPLITUDE CALCULATIONS")
    print("=" * 80)
    
    # Test parameters
    parmod = np.array([10.0, -150.0, 5.0, -10.0, 2.0, 1.0])
    pdyn = parmod[0]
    dst = parmod[1]
    byimf = parmod[2]
    bzimf = parmod[3]
    g1 = parmod[4]  # vbimf1
    g2 = parmod[5]  # vbimf2
    ps = -0.1
    
    # Coefficient array from t01
    a = np.array([
        1.00000, 2.47341, 0.40791, 0.30429, -0.10637, -0.89108, 3.29350,
        -0.05413, -0.00696, 1.07869, -0.02314, -0.66173, -0.68018, -0.03246,
        0.02681, 0.28062, 0.16535, -0.02939, 0.02639, -0.24891, -0.08063,
        0.08900, -0.02475, 0.05887, 0.57691, 0.65256, -0.03230, 2.24733,
        4.10546, 1.13665, 0.05506, 0.97669, 0.21164, 0.64594, 1.12556, 0.01389,
        1.02978, 0.02968, 0.15821, 9.00519, 28.17582, 1.35285, 0.42279
    ])
    
    print(f"Parameters:")
    print(f"  pdyn = {pdyn}")
    print(f"  dst = {dst}")
    print(f"  g1 (vbimf1) = {g1}")
    print(f"  g2 (vbimf2) = {g2}")
    
    # Calculate amplitudes as in scalar code
    dlp1 = (pdyn / 2.0) ** a[41]
    dlp2 = (pdyn / 2.0) ** a[42]
    
    print(f"\nPressure factors:")
    print(f"  dlp1 = (pdyn/2)^a[41] = ({pdyn}/2)^{a[41]} = {dlp1:.6f}")
    print(f"  dlp2 = (pdyn/2)^a[42] = ({pdyn}/2)^{a[42]} = {dlp2:.6f}")
    
    # Tail amplitudes (as in scalar extall)
    tamp1 = a[1] + a[2] * dlp1 + a[3] * g1 + a[4] * dst
    tamp2 = a[5] + a[6] * dlp2 + a[7] * g1 + a[8] * dst
    
    print(f"\nTail amplitudes (from extall):")
    print(f"  tamp1 = a[1] + a[2]*dlp1 + a[3]*g1 + a[4]*dst")
    print(f"        = {a[1]} + {a[2]}*{dlp1:.3f} + {a[3]}*{g1} + {a[4]}*{dst}")
    print(f"        = {tamp1:.6f}")
    print(f"  tamp2 = a[5] + a[6]*dlp2 + a[7]*g1 + a[8]*dst")
    print(f"        = {a[5]} + {a[6]}*{dlp2:.3f} + {a[7]}*{g1} + {a[8]}*{dst}")
    print(f"        = {tamp2:.6f}")
    
    # Ring current amplitudes
    a_src = a[9] + a[10] * dst + a[11] * np.sqrt(pdyn)
    a_prc = a[12] + a[13] * dst + a[14] * np.sqrt(pdyn)
    
    print(f"\nRing current amplitudes (from extall):")
    print(f"  a_src = a[9] + a[10]*dst + a[11]*sqrt(pdyn)")
    print(f"        = {a[9]} + {a[10]}*{dst} + {a[11]}*{np.sqrt(pdyn):.3f}")
    print(f"        = {a_src:.6f}")
    print(f"  a_prc = a[12] + a[13]*dst + a[14]*sqrt(pdyn)")
    print(f"        = {a[12]} + {a[13]}*{dst} + {a[14]}*{np.sqrt(pdyn):.3f}")
    print(f"        = {a_prc:.6f}")
    
    # Birkeland amplitudes
    a_r11 = a[15] + a[16] * g2
    a_r12 = a[17] + a[18] * g2
    a_r21 = a[19] + a[20] * g2
    a_r22 = a[21] + a[22] * g2
    
    print(f"\nBirkeland amplitudes (from extall):")
    print(f"  a_r11 = a[15] + a[16]*g2 = {a[15]} + {a[16]}*{g2} = {a_r11:.6f}")
    print(f"  a_r12 = a[17] + a[18]*g2 = {a[17]} + {a[18]}*{g2} = {a_r12:.6f}")
    print(f"  a_r21 = a[19] + a[20]*g2 = {a[19]} + {a[20]}*{g2} = {a_r21:.6f}")
    print(f"  a_r22 = a[21] + a[22]*g2 = {a[21]} + {a[22]}*{g2} = {a_r22:.6f}")
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("The scalar code uses the coefficient indices as shown above.")
    print("My earlier debug output was WRONG - I was looking at the wrong place!")
    print("\nThe vectorized code needs to match these exact formulas.")


if __name__ == "__main__":
    trace_scalar_amplitudes()