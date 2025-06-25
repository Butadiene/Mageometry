# T96 Vectorization Accuracy Issues Summary

## Test Point: (0, 5, 0) with parmod = [2.0, -10.0, 1.0, -5.0, 0, 0, 0, 0, 0, 0], ps = 0.0

### Main Issue: birk2tot_02 (Region 2 Birkeland Currents)

The vectorized birk2tot_02 function returns completely wrong values:
- Scalar: (0.000000, 0.000000, -0.204198)
- Vectorized: (0.358893, 0.027484, 0.348025)

Root cause: **loops4_vectorized** function has a bug in the coordinate transformation:

```python
# Incorrect (current implementation line 2057-2058):
xs = (x - xc) * cp + sign_y * (y - sign_y * yc) * sp
yss = sign_y * (y - sign_y * yc) * cp - (x - xc) * sp

# Should be (based on scalar implementation):
# For quadrant 1 (sign_y=1, sign_z=1):
xs = (x - xc) * cp + (y - yc) * sp
yss = (y - yc) * cp - (x - xc) * sp

# For quadrant 2 (sign_y=-1, sign_z=1):
xs = (x - xc) * cp - (y + yc) * sp
yss = (y + yc) * cp + (x - xc) * sp
```

Test results for loops4:
- Scalar: (0.000000, 0.000000, 0.121585)
- Vectorized: (-0.179602, -0.013754, -0.154751)

### Secondary Issue: tailrc96 Minor Discrepancies

Small differences in all three components:
- RC: Bz difference = -0.000315
- T2: Bz difference = 0.000097
- T3: Bz difference = -0.031760

These are much smaller errors (~0.03% for RC, ~0.01% for T2, ~19% for T3) but still contribute to overall inaccuracy.

### Impact on Total Field

At point (0, 5, 0):
- Scalar total: (0.106151, 0.361284, -39.705977) |B| = 39.707762
- Vectorized total: (5.607613, 0.782592, -31.467605) |B| = 31.972925
- Error in |B|: 19.48%

The main error comes from the Bx component (5182% error!) due to the birk2tot_02 bug.

### Fix Required

1. Fix loops4_vectorized coordinate transformation logic
2. Investigate minor discrepancies in tailrc96_vectorized components
3. Re-test accuracy across all test points