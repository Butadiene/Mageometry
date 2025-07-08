# Field Line Directional Derivatives Design

## DEPRECATED

This design document is deprecated. Please refer to `FIELD_LINE_DIRECTIONAL_DERIVATIVES_CORRECT.md` for the correct mathematical framework and implementation details.

The original design in this document contained fundamental misunderstandings about which directional derivative formulas to focus on. The correct implementation focuses on 9 specific formulas with proper antisymmetry relations:

1. (∂T/∂T)·n = κ (curvature)
2. (∂T/∂T)·b = 0
3. (∂n/∂T)·b = τ (torsion)
4. (∂T/∂n)·n
5. (∂T/∂n)·b
6. (∂n/∂n)·b
7. (∂n/∂b)·b
8. (∂n/∂b)·T
9. (∂b/∂b)·T

The correct implementation is in `geopack/vectorized/field_line_directional_derivatives_new.py`.

## Original Content (For Historical Reference Only)

[The original content has been preserved below for historical reference, but should not be used for implementation.]

---

[Original content follows...]