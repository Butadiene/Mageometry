Excellent work. You have successfully navigated the most complex structural challenges of the vectorization process. Completing the stabilization phase is a critical milestone, and having a fully integrated, high-performance model—even with known accuracy limitations—is a fantastic achievement.

The problem you are now facing, numerical instability in the full PRC implementation, is a classic and often final challenge in developing scientific computing models. You have moved from architectural problems to the nuanced domain of numerical precision and stability.

Here is a focused, strategic plan for what to do next. The absolute top priority is to solve the PRC numerical instability.

-----

### **Top Priority: The Numerical Instability Debugging Playbook**

The "extreme values" in your full PRC implementation are the key issue. The 7x error in storm conditions is a direct symptom of this. Before touching any other part of the code, you must diagnose and fix this instability. Here is a systematic playbook.

#### **Step 1: Isolate the Problem**

Focus exclusively on the problematic functions. Do not debug the entire T01 model. Your targets are:

  * `prc_quad_vectorized`
  * Its dependencies: `br_prc_q` and `bt_prc_q`

Create a new, separate test script (`debug_prc.py`) that calls these functions directly.

#### **Step 2: Term-by-Term Analysis (The Most Critical Step)**

The `br_prc_q` and `bt_prc_q` functions are a summation of many complex terms (`d1`, `d2`, ... `d18`). The instability is almost certainly coming from one or a few of these terms blowing up under certain conditions.

  * **Action:** Modify your vectorized `br_prc_q` and `bt_prc_q` functions to not just return the final sum, but to also return a dictionary or structured array containing the value of **each intermediate term**.

**Example for `br_prc_q`:**

```python
def debug_br_prc_q(r, sint, cost):
    # ... (calculate alpha, gamma, etc.) ...

    terms = {}
    
    # Calculate each term and store it
    f, fa, fs = ffs(alpha, al1, dal1)
    terms['d1'] = sc * f**xk1 / ((r/b1)**be1 + 1.)
    terms['d2'] = terms['d1'] * cost2
    
    f, fa, fs = ffs(alpha, al2, dal2)
    terms['d3'] = sc * fs**xk2 / ((r/b2)**be2 + 1.)
    terms['d4'] = terms['d3'] * cost2
    
    # ... continue for all 18 terms ...
    
    # Calculate the final sum
    final_br = a1*terms['d1'] + a2*terms['d2'] + a3*terms['d3'] # ... and so on
    
    return final_br, terms
```

#### **Step 3: Find the "Exploding" Term**

  * **Action:**
    1.  Identify a single input point (`x`, `y`, `z`) and parameter set (`parmod`) that causes the extreme values. Storm-time conditions are a good place to start.
    2.  Using your new `debug_..._q` functions, call them with this single point.
    3.  Print the values of all terms in the returned dictionary. You will likely see one or more terms with massive or `inf` values. This is your culprit.

#### **Step 4: Root Cause Analysis**

Once you know the unstable term (e.g., `d7`), examine its formula and the inputs that feed into it (`r`, `alpha`, `gamma`).

  * **Check for Catastrophic Cancellation:** Are you subtracting two very large, nearly equal numbers? This is a common source of instability.
  * **Check for Division by Zero (or near-zero):** Look at the denominators. For example, in `d7=sc/arga/argg`, what are the values of `arga` and `argg`? If they become extremely small, the term will explode.
  * **Compare with the Scalar Ground Truth:** Run the *original scalar code* for the same single input point. In the scalar code, print the values of `alpha`, `gamma`, and the individual terms (`d1`, `d2`, etc.). Compare them one-by-one to the values from your vectorized debug function.
      * If `alpha` or `gamma` are different, the error is in your coordinate transformations.
      * If `alpha` and `gamma` are identical but a term like `d7` is wildly different, the error is in your implementation of that specific formula or a subtle floating-point precision issue.

-----

### **Action Plan: From Debugging to Production**

#### **Priority 1: Fix the PRC Numerical Instability**

Follow the playbook above. This is your sole focus. The expected outcome is a set of `br_prc_q_vectorized` and `bt_prc_q_vectorized` functions that are numerically stable and match the scalar implementation to high precision for all inputs.

#### **Priority 2: Re-Implement and Validate the Full PRC**

Once the low-level functions are stable, re-integrate them into `prc_quad_vectorized` and `full_rc_vectorized`.

  * **Action:** Run your validation suite specifically for the ring current (`iopgen=4`).
  * **Goal:** The accuracy error should drop from `7x` to something well under 5-10% for the isolated ring current component, even during storms.

#### **Priority 3: Implement the Full Birkeland Current Model**

With the PRC model now stable and accurate, apply the same focused refinement to the Birkeland currents.

  * **Action:** Target the `fialcos` function. Replace the simplified version with the full, robust implementation using `np.select` as planned in the policy.
  * **Goal:** Improve storm-time accuracy further. After this step, your total field accuracy (`iopgen=0`) should be significantly improved.

#### **Priority 4: Final Validation and Documentation**

  * **Action:** Run your final, comprehensive end-to-end validation across the entire test suite. Document any remaining, small discrepancies and confirm that the model meets the final accuracy targets.
  * **Goal:** A fully validated, production-quality scientific model.

### **Sanity Check Table for Debugging**

When you are debugging, use a table like this to compare your vectorized output (`_v`) against the scalar ground truth (`_s`) for a single problematic point:

| Variable | Scalar Value (`_s`) | Vectorized Value (`_v`) | Difference |
| :--- | :--- | :--- | :--- |
| `r` | 7.51 | 7.51 | 0 |
| `sint` | 0.89 | 0.89 | 0 |
| `cost` | 0.45 | 0.45 | 0 |
| `alpha` | 0.105 | 0.105 | 0 |
| `gamma` | 0.0079 | 0.0079 | 0 |
| **`d1`** | -15.3 | -15.3 | 0 |
| **`d2`** | -3.1 | -3.1 | 0 |
| ... | ... | ... | ... |
| **`d7`** | **-4.7e+12** | **-4.7e+12** | **0** |
| `br_prc_q` | -4.7e+12 | -4.7e+12 | 0 |

This process will lead you directly to the source of the instability. You have a solid, stable foundation; now it's time for focused, methodical numerical detective work. You are on the final stretch to completing a truly production-quality tool.