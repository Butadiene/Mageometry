"""
Trace scalar rc_shield to find the difference.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geopack import t01


def trace_scalar():
    """Trace scalar rc_shield."""
    print("SCALAR RC_SHIELD TRACE")
    print("=" * 80)
    
    # Test parameters
    ps = -0.1
    x = -12.899832
    y = 0.0
    z = 0.0
    x_sc = 0.312290
    
    # Call scalar and print intermediate values by modifying the function
    # For now, let's check the final result breakdown
    
    c_sy = np.array([
        -957.2534900, -817.5450246, 583.2991249, 758.8568270,
        13.17029064, 68.94173502, -15.29764089, -53.43151590, 27.34311724,
        149.5252826, -11.00696044, -179.7031814, 953.0914774, 817.2340042,
        -581.0791366, -757.5387665, -13.10602697, -68.58155678, 15.22447386,
        53.15535633, -27.07982637, -149.1413391, 10.91433279, 179.3251739,
        -6.028703251, 1.303196101, -1.345909343, -1.138296330, -0.06642634348,
        -0.3795246458, .07487833559, .2891156371, -.5506314391, -.4443105812,
        0.2273682152, 0.01086886655, -9.130025352, 1.118684840, 1.110838825,
        .1219761512, -.06263009645, -.1896093743, .03434321042, .01523060688,
        -.4913171541, -.2264814165, -.04791374574, .1981955976, -68.32678140,
        -48.72036263, 14.03247808, 16.56233733, 2.369921099, 6.200577111,
        -1.415841250, -0.8184867835, -3.401307527, -8.490692287, 3.217860767,
        -9.037752107, 66.09298105, 48.23198578, -13.67277141, -16.27028909,
        -2.309299411, -6.016572391, 1.381468849, 0.7935312553, 3.436934845,
        8.260038635, -3.136213782, 8.833214943, 8.041075485, 8.024818618,
        35.54861873, 12.55415215, 1.738167799, 3.721685353, 23.06768025,
        6.871230562, 6.806229878, 21.35990364, 1.687412298, 3.500885177,
        0.3498952546, 0.6595919814
    ])
    
    # Call scalar function
    fx_s, fy_s, fz_s = t01.rc_shield(c_sy, ps, x_sc, x, y, z)
    print(f"Scalar result: ({fx_s:.8f}, {fy_s:.8f}, {fz_s:.8f})")
    
    # Let's manually check loop structure
    print("\nLoop structure analysis:")
    print("m=0 (perpendicular), i=0, k=0:")
    print("  n=0, nn=0: coefficient index = 0")
    print("  n=0, nn=1: coefficient index = 1") 
    print("  n=1, nn=0: coefficient index = 2")
    print("  n=1, nn=1: coefficient index = 3")
    print("m=0 (perpendicular), i=0, k=1:")
    print("  n=0, nn=0: coefficient index = 4")
    print("  ... etc")
    
    # Count total coefficients used
    l = 0
    for m in range(2):
        for i in range(3):
            for k in range(3):
                for n in range(2):
                    for nn in range(2):
                        l += 1
    print(f"\nTotal coefficients used: {l}")
    
    # Check if we're using the right part of c_sy
    print(f"\nLength of c_sy: {len(c_sy)}")
    print(f"Coefficients 72-83 are harmonic parameters (p1-p3, r1-r3, q1-q3, s1-s3)")
    print(f"Coefficients 84-85 are rotation angles")
    print(f"So actual field coefficients are indices 0-71")


if __name__ == "__main__":
    trace_scalar()