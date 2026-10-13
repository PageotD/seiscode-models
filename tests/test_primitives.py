"""Tests for primitive functions and dtype preservation."""
import numpy as np
import pytest

import seiscode_models as sm


class TestGradientBackground:
    def test_shape_and_dtype(self):
        v_f32 = sm.gradient_background(50, 30, dtype="f32")
        assert v_f32.shape == (50, 30)
        assert v_f32.dtype == np.float32

        v_f64 = sm.gradient_background(50, 30, dtype="f64")
        assert v_f64.dtype == np.float64

    def test_gradient_values(self):
        v = sm.gradient_background(10, 10, v0=1000.0, v1=3000.0)
        assert np.allclose(v[:, 0], 1000.0)
        assert np.allclose(v[:, -1], 3000.0)

class TestAddGaussianAnomaly:
    def test_dtype_preserved(self):
        v = sm.gradient_background(40, 40, dtype=np.float64)
        v_out = sm.add_gaussian_anomaly(v, 20, 20, 5.0, 5.0, 500.0)
        assert v_out.dtype == np.float64
        assert v_out.shape == (40, 40)

class TestValidation:
    def test_valid_model(self):
        v = sm.gradient_background(20, 20, dtype="f32")
        sm.validate_model(v) # Ne doit pas lever d'exception

    def test_invalid_dtype(self):
        v = sm.gradient_background(20, 20).astype(np.int32)
        with pytest.raises(ValueError, match="Model dtype must be float32 or float64"):
            sm.validate_model(v)
