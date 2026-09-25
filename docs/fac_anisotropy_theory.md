# Baseline theory: FAC and transverse magnetic anisotropy

[Documentation index](README.md) · [Transverse API](transverse_geometry.md)
· [Viewer guide](viewer.md)

This document defines the project's baseline for field-aligned current
(FAC) and transverse magnetic anisotropy. It specifies the notation,
equations, assumptions, and implementation without requiring an external
document. Numerical variables, API keys, and viewer controls use `beta_g`
and `delta_g` for **β_g** and **δ_g**. The subscript g distinguishes
geometrical shear from plasma beta.

The mathematical definitions and identities below are separate from the
R1/R2 interpretations and experiments in Section 7, which remain hypotheses
to test.

The literature connections are cited at the relevant equations below, with
version-specific references in Section 8. The component names `beta_g` and
`delta_g` and the normalized diagnostic η are this repository's convention;
they should not be attributed to the cited papers under those names.

## 1. Frame, derivatives, and current convention

Let B = |**B**| > 0, **T** = **B**/B, and let s increase along **T**.
Where κ = |∂_s **T**| > 0, define the Frenet normal
**n** = ∂_s **T**/κ and binormal **b** = **T** × **n**. Lowercase **b**
is the binormal, not the magnetic field. Directional derivatives are
∂_n = **n**·∇, ∂_b = **b**·∇, and ∂_s = **T**·∇.

Use the magnetostatic/quasistatic Ampère relation μ₀**J** = ∇×**B**,
neglecting displacement current, and j_∥ = **J**·**T**. The decomposition is

$$
\mu_0\mathbf J =
B\left[(\partial_n\mathbf T)\cdot\mathbf b
       +(\partial_b\mathbf n)\cdot\mathbf T\right]\mathbf T
+ (\partial_b B)\mathbf n
+ (B\kappa-\partial_n B)\mathbf b.
$$

Define the entries of the transverse direction gradient as

$$
M=\begin{pmatrix}a&q\\p&d\end{pmatrix},\qquad
a=\mathbf n\cdot\partial_n\mathbf T,\quad
q=\mathbf n\cdot\partial_b\mathbf T,\quad
p=\mathbf b\cdot\partial_n\mathbf T,\quad
d=\mathbf b\cdot\partial_b\mathbf T.
$$

Rows specify output components (n, b); columns specify derivative
directions (n, b). Differentiating **n**·**T** = 0 gives
(∂_b **n**)·**T** = −q. Consequently

$$
\alpha=p-q=\frac{\mu_0j_\parallel}{B},\qquad
\mathcal D=B\left[(\partial_n\mathbf T)\cdot\mathbf b
                  -(\partial_b\mathbf n)\cdot\mathbf T\right]
           =B(p+q).
$$

The two native parallel-current terms are **Bp** and **−Bq**. Their sum
is μ₀j_∥; their difference is **𝒟**, a signed shear diagnostic rather than
another component of current.

This α is the generalized force-free parameter of
[Tassev and Savcheva (2019), Eqs. (80)–(81)][tassev-savcheva]. Its definition
does not assume a force-free field.

## 2. Canonical symbols and their implementation

| Symbol | Definition | API key / implementation |
| --- | --- | --- |
| μ₀j_∥ | **T**·curl(**B**) = Bα | `fac` (Cartesian); `mu0J_T` (legacy Frenet reconstruction) |
| α | p−q = μ₀j_∥/B | `alpha` |
| β_g | p+q = 𝒟/B | `beta_g` |
| δ_g | a−d | `delta_g` |
| Γ | √(β_g²+δ_g²) | `gamma`; displayed as `Gamma` |
| θ | a+d = tr M = ∇·**T** | Internal transverse trace; no public output yet |
| 𝒟 | Bβ_g | Legacy `B_twist_diff` estimates this using frame derivatives |
| ω_c | sgn(α)√max(α²−Γ²,0)/2 | `omega_c` |
| η | (α²−Γ²)/(α²+Γ²) | `eta` |
| A | Γ/\|α\|, for α ≠ 0 | Derived in analysis; no built-in output |
| ψ_S | ½ atan2(β_g, δ_g), modulo π | Derived principal-axis angle; undefined at Γ = 0 |

All rates and κ have inverse-length units. 𝒟 has field/length units;
η and A are dimensionless, and ψ_S is an angle. These are local spatial
diagnostics, not rates of time evolution.

The symbol q denotes the matrix entry **n**·∂_b **T**. The numerical
finite-difference keyword `delta` is a step length (h), unrelated to δ_g.
ASCII `beta_g`, `delta_g`, `Gamma`, `omega_c`, and `eta` in viewer text
correspond to β_g, δ_g, Γ, ω_c, and η in equations.

`field_line_transverse_geometry` supplies exactly seven outputs:
`alpha`, `beta_g`, `delta_g`, `gamma`, `omega_c`, `eta`, and `curvature`.
The numerical implementation, viewer and CLI inputs, and colour-limit
keys use these same diagnostic names.
`B_twist_diff/B` and first-gradient `beta_g` converge to the same
definition where the frame is valid, but can differ at finite step size.
Viewer current conversion also scales the current-like 𝒟 diagnostic;
its label becomes `D / mu0` when displayed in current-density units.

## 3. Rotation, strain, and the role of the frame

The transverse gradient decomposes as

$$
M=\frac\theta2 I
+\frac12\begin{pmatrix}\delta_g&\beta_g\\\beta_g&-\delta_g\end{pmatrix}
+\frac\alpha2\begin{pmatrix}0&-1\\1&0\end{pmatrix}.
$$

These parts describe isotropic dilation, symmetric traceless strain, and
rotation, respectively. If ∇·**B** = 0, then θ = −∂_s ln B. Magnetic
flux conservation constrains the trace; it does not determine β_g or δ_g.

This is the transverse part of the unit-director gradient decomposition
reviewed by [Selinger (2019), Sec. II.1, Eqs. (3)–(8)][selinger]. On setting
the director to **T**, his splay is θ, twist is α, bend vector is −κ**n**,
and biaxial-splay tensor Δ is S_⊥ defined in Section 4. His gradient
∂_i n_j uses the transpose of our output-row convention. This correspondence
concerns geometry; it does not assign liquid-crystal elastic constants or
an Oseen–Frank free energy to a plasma. In particular, biaxial splay is not
itself the full saddle-splay divergence (his Eq. (20)).

For a transverse separation direction
**e**(φ) = cosφ **n** + sinφ **b**, the local angular rate is

$$
\Omega(\phi)=\mathbf T\cdot[\mathbf e\times((\mathbf e\cdot\nabla)\mathbf T)]
=\frac\alpha2+\frac{\beta_g}2\cos2\phi-\frac{\delta_g}2\sin2\phi.
$$

Thus Ω_n = p, Ω_b = −q, Ω_n+Ω_b = α, Ω_n−Ω_b = β_g,
and the azimuthal mean is α/2. The extreme rates are (α±Γ)/2.
FAC constrains the mean; β_g and δ_g describe its directional variation.
The directional rate and its azimuthal mean agree with
[Tassev and Savcheva (2019), Eqs. (83)–(86)][tassev-savcheva].
With ∂_s **n** = −κ**T** + τ**b**, the angle measured in the Frenet
frame satisfies **dφ/ds = Ω(φ)−τ**, where τ is the field-line torsion.
Do not identify α/2 with the mean
Frenet-frame angular derivative without this correction.

[Liu et al. (2015), Appendix C.1, Eqs. (C6)–(C9)][liu-et-al], separate the
geometric ribbon-twist density into current and symmetric-gradient terms.
Translating their c₃ to our notation gives

$$
\frac{c_3}{B}=\frac{\beta_g\cos2\phi-\delta_g\sin2\phi}{2},\qquad
\frac{d\mathcal T_g}{ds}=\frac{\Omega(\phi)}{2\pi}
=\frac{\alpha}{4\pi}+\frac{c_3}{2\pi B}
$$

in the infinitesimal-separation limit. Thus c₃/B equals β_g/2 for
**e** = **n**, but generally depends on the separation direction. Their
symmetric tensor differentiates **B**, whereas our S_⊥ differentiates
its direction after transverse projection. This distinction fixes the
factor of B and prevents identifying c₃ with the invariant Γ.

For a rotated transverse basis **n**′ = cosψ **n** + sinψ **b**,
**b**′ = −sinψ **n** + cosψ **b**,

$$
\begin{pmatrix}\delta_g'\\\beta_g'\end{pmatrix}
=\begin{pmatrix}\cos2\psi&\sin2\psi\\-\sin2\psi&\cos2\psi\end{pmatrix}
\begin{pmatrix}\delta_g\\\beta_g\end{pmatrix}.
$$

Γ is invariant under this change. In a resolved Frenet frame, β_g is
a meaningful component relative to the curvature normal, but does not
measure the entire anisotropy. The stretching direction satisfies
δ_g = Γ cos2ψ_S and β_g = Γ sin2ψ_S. A change in β_g can reflect
either a change in Γ or rotation of the principal axes.
Γ equals the eigenvalue gap of the symmetric part of M, corresponding to
the symmetrized squeeze rate in
[Tassev and Savcheva (2019), Sec. IV.5][tassev-savcheva].

Freezing local coefficients in a transverse frame with no added rotation,

$$
\lambda_\pm=\frac\theta2\pm\frac12\sqrt{\Gamma^2-\alpha^2}.
$$

The signed imaginary part gives ω_c as in
[Tassev and Savcheva (2019), Eq. (93)][tassev-savcheva]; their Eq. (95)
also sets ω_c = 0 for real eigenvalues. Their minimally rotating normal
basis is discussed in Sec. III.3.

After removing isotropic dilation, η > 0 identifies local rotational
behavior, η < 0 anisotropic stretching, and η = 0 their boundary,
including simple shear. These labels do not establish finite-distance
winding or a flux rope. The implementation extends ω_c by zero on the
noncoiling side. η is undefined at α = Γ = 0; its finite value can still
be unreliable when the denominator is near the derivative error level.

## 4. First-gradient evaluation and validity

Use G_ij = ∂B_i/∂x_j, P = I−**T****T**ᵀ, and

$$
L=\frac1BPGP,\qquad
S_\perp=\frac{L+L^T}{2}-\frac{\operatorname{tr}L}{2}P,\qquad
\Gamma=\sqrt{2\operatorname{tr}(S_\perp^2)}.
$$

The 3D Cartesian tensor L and its 2D transverse representation M must
not be confused with the raw magnetic gradient G. In a valid Frenet frame,

$$
\mathcal D=\mathbf b\cdot G\mathbf n+\mathbf n\cdot G\mathbf b,
\qquad B\delta_g=\mathbf n\cdot G\mathbf n-\mathbf b\cdot G\mathbf b.
$$

These formulas require no derivative of **n**. The implementation computes
α from **T**·curl(**B**)/B and Γ from S_⊥. They remain available on
straight nonzero field lines; β_g and δ_g require a resolved curvature
normal. Null fields and invalid derivative stencils produce NaN. Exact
zero is otherwise a valid result; do not substitute zero for undefined
geometry. Use step/grid convergence and problem-specific uncertainty masks
for weak curvature and weak α²+Γ². No universal physical noise floor is
inferred from a model or simulation.

## 5. Analytic comparisons and background geometry

For k > 0, representative local transverse maps are:

| M, written by rows | α | β_g | δ_g | Γ | η |
| --- | --- | --- | --- | --- | --- |
| (0, −k; k, 0) | 2k | 0 | 0 | 0 | +1 |
| (0, 0; 2k, 0) | 2k | 2k | 0 | 2k | 0 |
| (0, k; k, 0) | 0 | 2k | 0 | 2k | −1 |
| (k, 0; 0, −k) | 0 | 0 | 2k | 2k | −1 |

The first two have equal α and different neighboring-field geometry.
The last example has 𝒟 = 0 but nonzero anisotropy. These local examples
do not require zero curvature.

A current-free centered dipole provides a further baseline. For a dipole
aligned with z, write

$$
\mathbf B_{\rm dip}=\frac{m}{r^5}(3xz,\ 3yz,\ 2z^2-x^2-y^2),\qquad
r=\sqrt{x^2+y^2+z^2}>0,
$$

where m is a nonzero constant setting the field strength. With ϑ the
colatitude from the dipole axis, its local diagnostics satisfy

$$
\alpha_{\rm dip}=0,\qquad \beta_{g,{\rm dip}}=0,\qquad
\Gamma_{\rm dip}=\frac{3|\cos\vartheta|\sin^2\vartheta}
{r(1+3\cos^2\vartheta)^{3/2}}.
$$

The β_g statement applies where the Frenet normal is defined; on the
straight dipole axis β_g is undefined even though Γ = 0.

Consequently Γ is not exclusively an effect of FAC. Compare the total
field and a specified background. A scalar difference Γ_total−Γ_background
is not the norm of a background-subtracted strain tensor; specify a common
coordinate/basis convention and perform tensor subtraction when that is
the intended quantity.

For **B** = Σ_k **B**_k, fix **T**, **n**, **b** from the total field
and define 𝒟_k = **b**·G_k**n** + **n**·G_k**b**. Then
𝒟 = Σ_k 𝒟_k. Recomputing a separate frame for each source destroys
this additive interpretation. Γ is a tensor norm and is not additive.
Turning a model current source on/off is a sensitivity experiment, not a
guarantee of a new self-consistent equilibrium.

## 6. Along-line connections and finite-distance maps

The current-based twist integral is
$\mathcal T_w=(4\pi)^{-1}\int\alpha\,ds$.
[Liu et al. (2015), Eq. (7) and Appendix C.1][liu-et-al], explain why its
agreement with geometric ribbon twist requires proximity to the reference
axis and a negligible symmetric-gradient contribution. An azimuthal mean
at one point is not a winding count for a specified pair of field lines.

Under ∇·**J** = 0 and ∇·**B** = 0, writing
**J** = (α/μ₀)**B** + **J**_⊥ gives

$$
\partial_s\alpha=-\frac{\mu_0}{B}\nabla\cdot\mathbf J_\perp.
$$

This describes spatial connection between parallel and perpendicular
current. It does not by itself locate a temporal FAC generator or prove
energy conversion. Under isotropic-pressure static force balance,
**J**_⊥ = **B**×∇p/B²; pressure-tensor and inertia terms require their
own force-balance treatment outside that limit. Pressure p here is distinct
from the matrix entry p above.

For infinitesimal transverse separation in a frame without extra rotation,
dF/ds = M(s)F, F(s₀) = I. The finite map requires the ordered evolution
of M, not merely ∫Γ ds; changing principal directions can offset earlier
stretching. For perpendicular cross-sections and solenoidal **B**,
|det F| = B(s₀)/B(s). If evolving in the Frenet basis, subtract the
frame connection τ[[0, −1], [1, 0]] from M. Cross-section elongation
alone does not imply increased current density.
For the transverse-deviation evolution and the distinction between local
squeezing and finite squashing, see
[Tassev and Savcheva (2019), Secs. III–IV][tassev-savcheva].

## 7. R1/R2 research hypotheses and proposed comparisons

The cited works establish unit-vector geometry and solar magnetic-field
diagnostics; they do not validate the magnetospheric R1/R2 hypotheses here.

Region 1 (R1) and Region 2 (R2) denote large-scale FAC regions labelled
independently of the transverse diagnostics. A proposed research program
compares magnetic tubes connected to those regions. The definitions above
do not establish a rule that R1 is rotational and R2 is shear dominated.
Suggested comparisons are:

- Match local time, altitude, and |α|; compare (|α|, Γ), η, β_g, and ψ_S.
- Trace α, Γ, and ∂_s α to distinguish changing tube geometry from
  parallel/perpendicular current connections. Treat open lines up to their
  boundaries rather than requiring equatorial crossings.
- Separate background and current-source contributions in a fixed frame.
- Vary IMF By or dynamic pressure; distinguish strength changes in Γ
  from principal-axis rotation. Keep frame and field-direction conventions
  consistent across hemispheres.
- With suitable MHD/observational data, compare geometry with pressure
  gradients, **J**·**E**, magnetic-tension work, and propagation. A static
  magnetic model alone supplies neither energy balance nor causality.
- Constrain transverse gradients from vector magnetic data. FAC maps alone
  cannot recover the symmetric strain components; a one-dimensional sheet
  assumption can impose Γ ≈ |α| rather than discover it.

Γ measures anisotropy of the magnetic-direction gradient. It is distinct
from particle-pressure anisotropy and from the shape of a current-density
map. Validating the proposed interpretations requires independent model
or observational evidence.

The current code implements the local diagnostics and viewer comparison,
not R1/R2 labelling, source-resolved tensors, ψ_S/A outputs, current-closure
diagnostics, or finite-distance F integration. Extensions should state the
additional assumptions and tests rather than treating this research plan
as already validated behavior.

This baseline was developed from the supplied discussion
**枝分かれ · FAC異方性の意味**. The definitions and assumptions used by this
repository are recorded in full above.

## 8. References

Equation and section numbers above refer to these specific arXiv versions.

1. Jonathan V. Selinger (2019), *Interpretation of saddle-splay and the
   Oseen-Frank free energy in liquid crystals*,
   [arXiv:1901.06306v1][selinger].
2. Svetlin Tassev and Antonia Savcheva (2019), *Coiling and Squeezing:
   Properties of the Local Transverse Deviations of Magnetic Field Lines*,
   [arXiv:1901.00865v1][tassev-savcheva].
3. Rui Liu, Bernhard Kliem, Viacheslav S. Titov, Jun Chen, Yuming Wang,
   Haimin Wang, Chang Liu, Yan Xu, and Thomas Wiegelmann (2015), *Structure,
   Stability, and Evolution of Magnetic Flux Ropes from the Perspective of
   Magnetic Twist*, [arXiv:1512.02338v2][liu-et-al].

[selinger]: https://arxiv.org/html/1901.06306v1
[tassev-savcheva]: https://arxiv.org/html/1901.00865v1
[liu-et-al]: https://arxiv.org/html/1512.02338v2
