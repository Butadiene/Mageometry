"""
Example demonstrating the 9 field line directional derivatives.

This script shows how to calculate and interpret the 9 key formulas
for field line geometry directional derivatives.
"""

import numpy as np
import matplotlib.pyplot as plt
from geopack import recalc
from geopack.vectorized import t96_vectorized
from geopack.vectorized.field_line_directional_derivatives_new import (
    field_line_directional_derivatives_vectorized,
    verify_antisymmetry_relations,
    get_curvature_torsion_from_derivatives
)

# Set up time and model parameters
ut = 0.0  # 1970-01-01 00:00:00
ps = recalc(ut)

# T96 model parameters: [Pdyn, Dst, ByIMF, BzIMF, ...]
parmod = [2.0, -18.0, 2.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

print("Field Line Directional Derivatives - 9 Key Formulas")
print("=" * 60)

# Single point example
x, y, z = -5.0, 0.0, 0.0
print(f"\nAnalyzing point: ({x}, {y}, {z}) Re")

# Calculate all derivatives
derivatives = field_line_directional_derivatives_vectorized(
    t96_vectorized, parmod, ps, x, y, z
)

# Display the 9 key formulas
print("\nThe 9 Directional Derivative Formulas:")
print("-" * 40)
print("\n1. Tangential derivatives (Frenet-Serret):")
print(f"   (∂T/∂T)·n = {derivatives['dT_dT_n']:8.5f}  (curvature κ)")
print(f"   (∂T/∂T)·b = {derivatives['dT_dT_b']:8.5f}  (should be 0)")
print(f"   (∂n/∂T)·b = {derivatives['dn_dT_b']:8.5f}  (torsion τ)")

print("\n2. Normal derivatives:")
print(f"   (∂T/∂n)·n = {derivatives['dT_dn_n']:8.5f}")
print(f"   (∂T/∂n)·b = {derivatives['dT_dn_b']:8.5f}")
print(f"   (∂n/∂n)·b = {derivatives['dn_dn_b']:8.5f}")

print("\n3. Binormal derivatives:")
print(f"   (∂n/∂b)·b = {derivatives['dn_db_b']:8.5f}")
print(f"   (∂n/∂b)·T = {derivatives['dn_db_T']:8.5f}")
print(f"   (∂b/∂b)·T = {derivatives['db_db_T']:8.5f}")

# Verify antisymmetry relations
print("\nVerifying Antisymmetry Relations:")
print("-" * 40)
errors = verify_antisymmetry_relations(derivatives)

for name, error in errors.items():
    status = "✓ PASS" if abs(error) < 1e-4 else "✗ FAIL"
    print(f"{name:20} error = {error:10.2e}  {status}")

# Array calculation along a line
print("\n\nAnalyzing variation along equatorial line:")
print("=" * 60)

x_line = np.linspace(-10, -3, 100)
y_line = np.zeros_like(x_line)
z_line = np.zeros_like(x_line)

# Calculate derivatives along line
derivatives_line = field_line_directional_derivatives_vectorized(
    t96_vectorized, parmod, ps, x_line, y_line, z_line
)

# Extract curvature and torsion
curvature, torsion = get_curvature_torsion_from_derivatives(derivatives_line)

# Create visualization
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('The 9 Directional Derivative Formulas Along Equatorial Line', fontsize=16)

# Plot each formula
plots = [
    (derivatives_line['dT_dT_n'], '(∂T/∂T)·n = κ', 'Curvature'),
    (derivatives_line['dT_dT_b'], '(∂T/∂T)·b', 'Should be ~0'),
    (derivatives_line['dn_dT_b'], '(∂n/∂T)·b = τ', 'Torsion'),
    (derivatives_line['dT_dn_n'], '(∂T/∂n)·n', 'Normal deriv 1'),
    (derivatives_line['dT_dn_b'], '(∂T/∂n)·b', 'Normal deriv 2'),
    (derivatives_line['dn_dn_b'], '(∂n/∂n)·b', 'Normal deriv 3'),
    (derivatives_line['dn_db_b'], '(∂n/∂b)·b', 'Binormal deriv 1'),
    (derivatives_line['dn_db_T'], '(∂n/∂b)·T', 'Binormal deriv 2'),
    (derivatives_line['db_db_T'], '(∂b/∂b)·T', 'Binormal deriv 3')
]

for idx, (data, formula, title) in enumerate(plots):
    ax = axes[idx // 3, idx % 3]
    ax.plot(x_line, data, 'b-', linewidth=2)
    ax.set_xlabel('X (Re)')
    ax.set_ylabel('Value')
    ax.set_title(f'{formula}\n{title}')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('field_line_9_derivatives.png', dpi=150)
plt.show()

# Create antisymmetry validation plot
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Antisymmetry Relations Validation', fontsize=16)

# Calculate all antisymmetry errors
errors_line = verify_antisymmetry_relations(derivatives_line)

# Plot each antisymmetry check
antisym_plots = [
    (derivatives_line['dT_dT_n'] + derivatives_line['dn_dT_T'], 
     '(∂T/∂T)·n + (∂n/∂T)·T', 'Should be 0'),
    (derivatives_line['dT_dT_b'] - derivatives_line['db_dT_T'], 
     '(∂T/∂T)·b - (∂b/∂T)·T', 'Should be 0'),
    (derivatives_line['dn_dT_b'] + derivatives_line['db_dT_n'], 
     '(∂n/∂T)·b + (∂b/∂T)·n', 'Should be 0'),
    (derivatives_line['dT_dn_n'] + derivatives_line['dn_dn_T'], 
     '(∂T/∂n)·n + (∂n/∂n)·T', 'Should be 0'),
    (derivatives_line['dT_dn_b'] + derivatives_line['db_dn_T'], 
     '(∂T/∂n)·b + (∂b/∂n)·T', 'Should be 0'),
    (derivatives_line['dn_dn_b'] + derivatives_line['db_dn_n'], 
     '(∂n/∂n)·b + (∂b/∂n)·n', 'Should be 0'),
    (derivatives_line['dn_db_b'] + derivatives_line['db_db_n'], 
     '(∂n/∂b)·b + (∂b/∂b)·n', 'Should be 0'),
    (derivatives_line['dn_db_T'] + derivatives_line['dT_db_n'], 
     '(∂n/∂b)·T + (∂T/∂b)·n', 'Should be 0'),
    (derivatives_line['db_db_T'] + derivatives_line['dT_db_b'], 
     '(∂b/∂b)·T + (∂T/∂b)·b', 'Should be 0')
]

for idx, (data, formula, title) in enumerate(antisym_plots):
    ax = axes[idx // 3, idx % 3]
    ax.plot(x_line, data, 'r-', linewidth=2)
    ax.set_xlabel('X (Re)')
    ax.set_ylabel('Error')
    ax.set_title(f'{formula}\n{title}')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Add text with max error
    max_error = np.max(np.abs(data))
    ax.text(0.02, 0.98, f'Max error: {max_error:.2e}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('field_line_antisymmetry_validation.png', dpi=150)
plt.show()

# Summary statistics
print("\nSummary Statistics:")
print("=" * 60)
print(f"Curvature κ range: {curvature.min():.5f} to {curvature.max():.5f} 1/Re")
print(f"Torsion τ range: {torsion.min():.5f} to {torsion.max():.5f} 1/Re")

print("\nMaximum antisymmetry errors:")
for name, error_array in errors_line.items():
    max_error = np.max(np.abs(error_array))
    print(f"  {name:20} = {max_error:.2e}")

print("\nPhysical interpretation:")
print("- Curvature κ: How rapidly the field line bends")
print("- Torsion τ: How the osculating plane rotates")
print("- Normal derivatives: Field line bundle divergence/convergence")
print("- Binormal derivatives: Out-of-plane geometric properties")