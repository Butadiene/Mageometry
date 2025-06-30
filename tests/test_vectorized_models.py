#!/usr/bin/env python
"""
Consolidated tests for all vectorized model implementations.

This module tests the accuracy and performance of vectorized implementations
against their scalar counterparts for all magnetospheric models.
"""

import numpy as np
import pytest
import time
import sys
import os

# Add parent directory to path

import geopack
from geopack import t89, t96, t01, t04
from geopack import t89_vectorized, t96_vectorized, t01_vectorized, t04_vectorized


class TestVectorizedAccuracy:
    """Test accuracy of vectorized implementations against scalar versions."""
    
    @classmethod
    def setup_class(cls):
        """Set up test parameters."""
        # Calculate dipole tilt
        import datetime
        dt = datetime.datetime(2023, 3, 15, 12, 0, 0)
        ut = dt.timestamp()
        cls.ps = geopack.recalc(ut)
        
        # T89 parameters
        cls.kp = 3
        
        # T96 parameters
        cls.parmod_t96 = np.array([2.0, -20.0, 0.0, -5.0, 0, 0, 0, 0, 0, 0])
        
        # T01 parameters
        cls.parmod_t01 = np.array([2.0, -30.0, 2.0, -5.0, 0.5, 1.0, 0, 0, 0, 0])
        
        # T04 parameters
        cls.parmod_t04 = np.array([5.0, -50.0, 2.0, -5.0, 0.5, 1.0, 0.8, 1.2, 0.6, 0.9])
    
    def test_t89_accuracy(self):
        """Test T89 vectorized accuracy."""
        n_test = 100
        x = np.random.uniform(-20, 10, n_test)
        y = np.random.uniform(-10, 10, n_test)
        z = np.random.uniform(-5, 5, n_test)
        
        errors = []
        for i in range(n_test):
            bx_s, by_s, bz_s = t89(self.kp, self.ps, x[i], y[i], z[i])
            bx_v, by_v, bz_v = t89_vectorized(self.kp, self.ps, x[i], y[i], z[i])
            
            b_mag_s = np.sqrt(bx_s**2 + by_s**2 + bz_s**2)
            if b_mag_s > 1e-10:
                error = np.sqrt((bx_v-bx_s)**2 + (by_v-by_s)**2 + (bz_v-bz_s)**2) / b_mag_s
                errors.append(error)
        
        errors = np.array(errors)
        assert np.max(errors) < 1e-10, f"T89 max error: {np.max(errors)}"
        assert np.mean(errors) < 1e-12, f"T89 mean error: {np.mean(errors)}"
    
    def test_t96_accuracy(self):
        """Test T96 vectorized accuracy."""
        n_test = 100
        x = np.random.uniform(-20, 10, n_test)
        y = np.random.uniform(-10, 10, n_test)
        z = np.random.uniform(-5, 5, n_test)
        
        errors = []
        for i in range(n_test):
            bx_s, by_s, bz_s = t96(self.parmod_t96, self.ps, x[i], y[i], z[i])
            bx_v, by_v, bz_v = t96_vectorized(self.parmod_t96, self.ps, x[i], y[i], z[i])
            
            b_mag_s = np.sqrt(bx_s**2 + by_s**2 + bz_s**2)
            if b_mag_s > 1e-10:
                error = np.sqrt((bx_v-bx_s)**2 + (by_v-by_s)**2 + (bz_v-bz_s)**2) / b_mag_s
                errors.append(error)
        
        errors = np.array(errors)
        assert np.max(errors) < 1e-6, f"T96 max error: {np.max(errors)}"
        assert np.mean(errors) < 1e-8, f"T96 mean error: {np.mean(errors)}"
    
    def test_t01_accuracy(self):
        """Test T01 vectorized accuracy."""
        n_test = 100
        x = np.random.uniform(-15, 10, n_test)
        y = np.random.uniform(-10, 10, n_test)
        z = np.random.uniform(-5, 5, n_test)
        
        errors = []
        for i in range(n_test):
            bx_s, by_s, bz_s = t01(self.parmod_t01, self.ps, x[i], y[i], z[i])
            bx_v, by_v, bz_v = t01_vectorized(self.parmod_t01, self.ps, x[i], y[i], z[i])
            
            b_mag_s = np.sqrt(bx_s**2 + by_s**2 + bz_s**2)
            if b_mag_s > 1e-10:
                error = np.sqrt((bx_v-bx_s)**2 + (by_v-by_s)**2 + (bz_v-bz_s)**2) / b_mag_s
                errors.append(error)
        
        errors = np.array(errors)
        assert np.max(errors) < 1e-8, f"T01 max error: {np.max(errors)}"
        assert np.mean(errors) < 1e-10, f"T01 mean error: {np.mean(errors)}"
    
    def test_t04_accuracy(self):
        """Test T04 vectorized accuracy."""
        n_test = 100
        x = np.random.uniform(-10, 5, n_test)  # T04 valid for X > -15
        y = np.random.uniform(-10, 10, n_test)
        z = np.random.uniform(-5, 5, n_test)
        
        errors = []
        for i in range(n_test):
            bx_s, by_s, bz_s = t04(self.parmod_t04, self.ps, x[i], y[i], z[i])
            bx_v, by_v, bz_v = t04_vectorized(self.parmod_t04, self.ps, x[i], y[i], z[i])
            
            b_mag_s = np.sqrt(bx_s**2 + by_s**2 + bz_s**2)
            if b_mag_s > 1e-10:
                error = np.sqrt((bx_v-bx_s)**2 + (by_v-by_s)**2 + (bz_v-bz_s)**2) / b_mag_s
                errors.append(error)
        
        errors = np.array(errors)
        assert np.max(errors) < 1e-8, f"T04 max error: {np.max(errors)}"
        assert np.mean(errors) < 1e-10, f"T04 mean error: {np.mean(errors)}"


class TestVectorizedPerformance:
    """Test performance improvements of vectorized implementations."""
    
    @classmethod
    def setup_class(cls):
        """Set up test parameters."""
        TestVectorizedAccuracy.setup_class()
        cls.ps = TestVectorizedAccuracy.ps
        cls.kp = TestVectorizedAccuracy.kp
        cls.parmod_t96 = TestVectorizedAccuracy.parmod_t96
        cls.parmod_t01 = TestVectorizedAccuracy.parmod_t01
        cls.parmod_t04 = TestVectorizedAccuracy.parmod_t04
    
    def test_t89_performance(self):
        """Test T89 vectorized performance."""
        n_points = 1000
        x = np.random.uniform(-20, 10, n_points)
        y = np.random.uniform(-10, 10, n_points)
        z = np.random.uniform(-5, 5, n_points)
        
        # Time scalar
        t0 = time.time()
        for i in range(100):
            _ = t89(self.kp, self.ps, x[i], y[i], z[i])
        t_scalar = (time.time() - t0) * n_points / 100
        
        # Time vectorized
        t0 = time.time()
        _ = t89_vectorized(self.kp, self.ps, x, y, z)
        t_vector = time.time() - t0
        
        speedup = t_scalar / t_vector
        assert speedup > 10, f"T89 speedup only {speedup:.1f}x, expected > 10x"
    
    def test_t96_performance(self):
        """Test T96 vectorized performance."""
        n_points = 1000
        x = np.random.uniform(-20, 10, n_points)
        y = np.random.uniform(-10, 10, n_points)
        z = np.random.uniform(-5, 5, n_points)
        
        # Time scalar
        t0 = time.time()
        for i in range(100):
            _ = t96(self.parmod_t96, self.ps, x[i], y[i], z[i])
        t_scalar = (time.time() - t0) * n_points / 100
        
        # Time vectorized
        t0 = time.time()
        _ = t96_vectorized(self.parmod_t96, self.ps, x, y, z)
        t_vector = time.time() - t0
        
        speedup = t_scalar / t_vector
        assert speedup > 10, f"T96 speedup only {speedup:.1f}x, expected > 10x"


class TestVectorizedInterface:
    """Test interface compatibility of vectorized implementations."""
    
    @classmethod
    def setup_class(cls):
        """Set up test parameters."""
        TestVectorizedAccuracy.setup_class()
        cls.ps = TestVectorizedAccuracy.ps
        cls.kp = TestVectorizedAccuracy.kp
        cls.parmod_t96 = TestVectorizedAccuracy.parmod_t96
    
    def test_scalar_input_output(self):
        """Test that vectorized functions handle scalar inputs correctly."""
        x, y, z = 5.0, 0.0, 0.0
        
        # T89
        bx, by, bz = t89_vectorized(self.kp, self.ps, x, y, z)
        assert np.isscalar(bx) and np.isscalar(by) and np.isscalar(bz)
        
        # T96
        bx, by, bz = t96_vectorized(self.parmod_t96, self.ps, x, y, z)
        assert np.isscalar(bx) and np.isscalar(by) and np.isscalar(bz)
    
    def test_array_shape_preservation(self):
        """Test that output shape matches input shape."""
        shapes = [(10,), (5, 4), (3, 3, 3)]
        
        for shape in shapes:
            x = np.random.uniform(-10, 10, shape)
            y = np.random.uniform(-10, 10, shape)
            z = np.random.uniform(-5, 5, shape)
            
            # T89
            bx, by, bz = t89_vectorized(self.kp, self.ps, x, y, z)
            assert bx.shape == shape
            assert by.shape == shape
            assert bz.shape == shape


if __name__ == "__main__":
    # Run tests
    print("Running vectorized model tests...")
    pytest.main([__file__, "-v"])