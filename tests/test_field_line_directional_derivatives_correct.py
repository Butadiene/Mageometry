"""
Test suite for the corrected field line directional derivatives implementation.

Tests the 9 directional derivative formulas with proper antisymmetry relations.
"""

import numpy as np
import unittest
from geopack import recalc
from geopack.vectorized import t96_vectorized
from geopack.vectorized.field_line_directional_derivatives_new import (
    field_line_directional_derivatives_vectorized,
    verify_antisymmetry_relations,
    get_curvature_torsion_from_derivatives
)
from geopack.vectorized.field_line_geometry_vectorized import (
    field_line_curvature_vectorized,
    field_line_torsion_vectorized
)


class TestFieldLineDirectionalDerivativesCorrect(unittest.TestCase):
    """Test the 9 directional derivative formulas."""
    
    def setUp(self):
        """Set up test parameters."""
        # Time for calculations
        self.ut = 0.0
        self.ps = recalc(self.ut)
        
        # Model parameters for T96
        self.parmod = [2.0, -18.0, 2.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Test positions
        self.x_scalar = -5.0
        self.y_scalar = 0.0
        self.z_scalar = 0.0
        
        self.x_array = np.array([-5.0, -6.0, -7.0, -8.0])
        self.y_array = np.array([0.0, 1.0, 0.0, -1.0])
        self.z_array = np.array([0.0, 0.0, 1.0, 0.0])
    
    def test_scalar_input(self):
        """Test that scalar inputs work correctly."""
        derivatives = field_line_directional_derivatives_vectorized(
            t96_vectorized, self.parmod, self.ps, 
            self.x_scalar, self.y_scalar, self.z_scalar
        )
        
        # Check that we get all 9 primary values
        expected_keys = [
            'dT_dT_n', 'dT_dT_b', 'dn_dT_b',
            'dT_dn_n', 'dT_dn_b', 'dn_dn_b',
            'dn_db_b', 'dn_db_T', 'db_db_T'
        ]
        
        for key in expected_keys:
            self.assertIn(key, derivatives)
            self.assertIsInstance(derivatives[key], (int, float))
    
    def test_array_input(self):
        """Test that array inputs return arrays."""
        derivatives = field_line_directional_derivatives_vectorized(
            t96_vectorized, self.parmod, self.ps,
            self.x_array, self.y_array, self.z_array
        )
        
        for key, value in derivatives.items():
            if key in ['dT_dT_n', 'dT_dT_b', 'dn_dT_b']:  # Main 9 formulas
                self.assertIsInstance(value, np.ndarray)
                self.assertEqual(value.shape, self.x_array.shape)
    
    def test_frenet_serret_formulas(self):
        """Test that Frenet-Serret formulas are satisfied."""
        derivatives = field_line_directional_derivatives_vectorized(
            t96_vectorized, self.parmod, self.ps,
            self.x_scalar, self.y_scalar, self.z_scalar
        )
        
        # Get curvature and torsion from standard functions
        curvature_expected = field_line_curvature_vectorized(
            t96_vectorized, self.parmod, self.ps,
            self.x_scalar, self.y_scalar, self.z_scalar
        )
        torsion_expected = field_line_torsion_vectorized(
            t96_vectorized, self.parmod, self.ps,
            self.x_scalar, self.y_scalar, self.z_scalar
        )
        
        # Extract from derivatives
        curvature, torsion = get_curvature_torsion_from_derivatives(derivatives)
        
        # Check Frenet-Serret formulas
        self.assertAlmostEqual(curvature, derivatives['dT_dT_n'], places=6)
        self.assertAlmostEqual(torsion, derivatives['dn_dT_b'], places=6)
        
        # (∂T/∂T)·b should be zero
        self.assertAlmostEqual(derivatives['dT_dT_b'], 0.0, places=6)
        
        # Compare with standard functions (allowing for numerical differences)
        self.assertAlmostEqual(curvature, curvature_expected, places=4)
        self.assertAlmostEqual(torsion, torsion_expected, places=4)
    
    def test_antisymmetry_relations(self):
        """Test all antisymmetry relations."""
        derivatives = field_line_directional_derivatives_vectorized(
            t96_vectorized, self.parmod, self.ps,
            self.x_scalar, self.y_scalar, self.z_scalar
        )
        
        errors = verify_antisymmetry_relations(derivatives)
        
        # All antisymmetry errors should be small
        for name, error in errors.items():
            self.assertLess(abs(error), 1e-4, 
                          f"Antisymmetry relation {name} failed: error = {error}")
    
    def test_zero_formulas(self):
        """Test that certain combinations are zero."""
        # For a simple test, use multiple points
        derivatives = field_line_directional_derivatives_vectorized(
            t96_vectorized, self.parmod, self.ps,
            self.x_array, self.y_array, self.z_array
        )
        
        # (∂T/∂T)·b should be zero
        np.testing.assert_allclose(derivatives['dT_dT_b'], 0.0, atol=1e-6)
        
        # (∂b/∂T)·T should be zero (from antisymmetry)
        np.testing.assert_allclose(derivatives['db_dT_T'], 0.0, atol=1e-6)
    
    def test_dipole_field(self):
        """Test with pure dipole field where we know some properties."""
        from geopack import dip
        
        def dipole_wrapper(parmod, ps, x, y, z):
            return dip(x, y, z)
        
        # Test at equatorial point
        x_eq = -5.0
        y_eq = 0.0
        z_eq = 0.0
        
        derivatives = field_line_directional_derivatives_vectorized(
            dipole_wrapper, None, 0.0, x_eq, y_eq, z_eq
        )
        
        # For dipole at equator, κ = 2/r
        r = np.sqrt(x_eq**2 + y_eq**2 + z_eq**2)
        expected_curvature = 2.0 / r
        
        self.assertAlmostEqual(derivatives['dT_dT_n'], expected_curvature, places=2)
        
        # Torsion should be near zero for dipole in meridional plane
        self.assertLess(abs(derivatives['dn_dT_b']), 1e-3)
    
    def test_vectorization_consistency(self):
        """Test that vectorized and scalar calculations match."""
        # Calculate for each point individually
        scalar_results = []
        for i in range(len(self.x_array)):
            deriv = field_line_directional_derivatives_vectorized(
                t96_vectorized, self.parmod, self.ps,
                self.x_array[i], self.y_array[i], self.z_array[i]
            )
            scalar_results.append(deriv)
        
        # Calculate for all points at once
        vector_results = field_line_directional_derivatives_vectorized(
            t96_vectorized, self.parmod, self.ps,
            self.x_array, self.y_array, self.z_array
        )
        
        # Compare key values
        for key in ['dT_dT_n', 'dn_dT_b', 'dT_dn_n']:
            scalar_vals = np.array([sr[key] for sr in scalar_results])
            vector_vals = vector_results[key]
            np.testing.assert_allclose(scalar_vals, vector_vals, rtol=1e-12)
    
    def test_antisymmetry_array(self):
        """Test antisymmetry relations for array inputs."""
        derivatives = field_line_directional_derivatives_vectorized(
            t96_vectorized, self.parmod, self.ps,
            self.x_array, self.y_array, self.z_array
        )
        
        # Check key antisymmetry relations
        # (∂T/∂T)·n = -(∂n/∂T)·T
        sum1 = derivatives['dT_dT_n'] + derivatives['dn_dT_T']
        np.testing.assert_allclose(sum1, 0.0, atol=1e-4)
        
        # (∂n/∂T)·b = -(∂b/∂T)·n
        sum2 = derivatives['dn_dT_b'] + derivatives['db_dT_n']
        np.testing.assert_allclose(sum2, 0.0, atol=1e-4)
        
        # (∂T/∂n)·n = -(∂n/∂n)·T
        sum3 = derivatives['dT_dn_n'] + derivatives['dn_dn_T']
        np.testing.assert_allclose(sum3, 0.0, atol=1e-4)


if __name__ == '__main__':
    unittest.main()