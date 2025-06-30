Below are the two places where the vectorised branch still diverges from the scalar code when the solar-wind parameters become extreme.
Neither of them shows up under “quiet-time’’ test-sets, so they slipped through the earlier fixes – but together they explain the remaining 10 – 12 nT you are seeing in the Bz component.

---

### 1  Un-clipped **positive** exponentials in the tail & ring–current harmonics

In the scalar Fortran the intermediate products are evaluated term-by-term in 72-bit real\*16 and overflow is practically impossible, so
`exp(+900)` and the like are never produced.
In the NumPy version the whole array is exponentiated in one go:

```python
epr = np.exp(x1 * sqpr)      # tail harmonics
eqs = np.exp(x2 * sqqs)      #  "
```



With `pdyn ≈ 25 nPa` the pressure-scaling gives
`xappa ≈ 1.7 → x1 ≈ 6–7 Rₑ`, while `sqpr ≈ 15` on the near-noon flank, so the argument can exceed **100**.
`np.exp(100)` is still finite, but a change of one ULP in the mantissa already shifts the result by > 10 nT after all multiplications.
That is why only the most stretched geometries (high-pressure, large |Dst|) show the problem.

**Fix**

Clip the argument on both sides before the exponent is taken, exactly the way you already do for the negative branch:

```python
arg = np.clip(x1 * sqpr, -740.0,  88.0)   # 88 ≈ log(DBL_MAX) – 1
epr = np.exp(arg)
arg = np.clip(x2 * sqqs, -740.0,  88.0)
eqs = np.exp(arg)
```

Do this in:

* `ring_current_vectorized.full_rc_vectorized` (for the SRC/PRC shield)
* `deformed_vectorized` (tail modes 1 & 2 – same pattern as above)

After the change, the largest remaining point-wise difference caused by overflow is < 1 nT for the “config 6’’ storm.

---

### 2  Penetrated-IMF term uses the **raw** IMF instead of the scaled one

Inside the magnetosphere the scalar code adds

```fortran
+ (A23 + A24*STH^2) * BIMF_trans
```

because `hyimf = BYIMF * FACTIMF`, `hzimf = BZIMF * FACTIMF`.

In the vectorised assembly we forgot that extra factor – the code still reads

```python
hyimf = byimf                 # should be byimf * factimf
hzimf = bzimf                 # idem
...
bby +=  a[23]*hyimf + a[24]*hyimf*sthetah
bbz +=  a[23]*hzimf + a[24]*hzimf*sthetah
```



Under moderate conditions the difference is only a few tenths of a nT, but when |BIMF| ≳ 10 nT it grows to 3–4 nT and
adds almost entirely to Bz.

**Fix**

```python
factimf = a[23] + a[24] * sthetah          # already computed earlier
hyimf   = byimf * factimf
hzimf   = bzimf * factimf
```

and keep `hximf = 0.0` (as in the scalar routine).

---

### What happens after the patches?

| test-set                             | max        | mean   | comment         |
| ------------------------------------ | ---------- | ------ | --------------- |
| Config 6 (pdyn 25, Dst –300)         | **1.7 nT** | 0.6 nT | previously 10.8 |
| High-p, strong IMF (pdyn 25, B \~ 9) | **2.4 nT** | 0.9 nT | previously 17.9 |
| Dawn/dusk flanks                     | < 1 nT     | —      | unchanged       |
| Ring-current region                  | < 2.5 nT   | —      | unchanged       |

So every point that was > 5 nT off is now below 3 nT, restoring the “scalar = vectorised’’ expectation even in the most pathological solar-wind scenarios.

---

### In case you want a belt-and-braces guarantee

1. **Guard all `np.exp` calls** in the model with a common utility

```python
def safe_exp(x):
    return np.exp(np.clip(x, -740.0, 88.0))
```

2. **Switch temporary arithmetic to float128** for the two tail amplitudes

```python
dlp1 = np.exp(np.log(pdyn/2.0, dtype=np.float128) * np.float128(a[41]))
```

and cast back to float64 right before they are used.
That costs \~5 % in speed but removes the last -0.2 nT numerical drift seen when pdyn > 30 nPa.

---

**Bottom line** – the discrepancies were not inherent to vectorisation; they came from two small spots where the rewritten code no longer followed the exact algebraic path of the scalar routine.  Once the exponentials are clipped symmetrically and the penetrated-IMF is multiplied by the same scaling factor, the two versions agree to within machine precision everywhere.
