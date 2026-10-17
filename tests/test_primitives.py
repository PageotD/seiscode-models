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


class TestAddEllipticalAnomaly:
    def test_elliptical_anomaly(self):
        v = sm.gradient_background(50, 50, dtype=np.float32)
        v_out = sm.add_elliptical_inclusion(v, 25, 25, 10.0, 5.0, 100.0)

        assert v_out.shape == (50, 50)
        assert v_out.dtype == v.dtype

        # Centre de l'ellipse.
        assert v_out[25, 25] == 100.0  # pytest.approx(v[25, 25] + 100.0)

        # Point à l'intérieur de l'ellipse.
        assert v_out[30, 25] == 100.0  # pytest.approx(v[30, 25] + 100.0)

        # Point à l'extérieur de l'ellipse.
        assert v_out[25, 35] == pytest.approx(v[25, 35])

        # Le modèle initial reste inchangé.
        assert v[25, 25] == pytest.approx(1500.0 + 3000.0 * 25 / 49)

    def test_elliptical_shape(self):
        v = np.zeros((50, 50), dtype=np.float32)
        v_out = sm.add_elliptical_inclusion(v, 25, 25, 10.0, 5.0, 100.0)

        # Étendue horizontale : demi-axe a = 10.
        assert v_out[33, 25] == pytest.approx(100.0)
        assert v_out[36, 25] == pytest.approx(0.0)

        # Étendue verticale : demi-axe b = 5.
        assert v_out[25, 29] == pytest.approx(100.0)
        assert v_out[25, 31] == pytest.approx(0.0)

    def test_dtype_preserved(self):
        v = sm.gradient_background(40, 40, dtype=np.float64)
        v_out = sm.add_elliptical_inclusion(v, 20, 20, 10.0, 5.0, 500.0)

        assert v_out.dtype == np.float64
        assert v_out.shape == (40, 40)

    def test_physical_coordinates(self):
        v = np.zeros((40, 40), dtype=np.float32)
        # dx=5.0m, dz=2.0m: center at x=50m (col 10), z=40m (row 20)
        v_out = sm.add_elliptical_inclusion(
            v, cx=50.0, cz=40.0, radius_x=15.0, radius_z=6.0, cv=3000.0, dx=5.0, dz=2.0
        )
        assert v_out[10, 20] == 3000.0
        assert v_out[0, 0] == 0.0

    def test_fault_offset_physical_coordinates(self):
        v = np.arange(20, dtype=np.float32)[None, :].repeat(20, axis=0)
        v_shifted = sm.add_fault_offset(v, x_fault=20.0, throw=6.0, dx=2.0, dz=2.0)
        # col_fault = 20 / 2 = 10, shift = 6 / 2 = 3 cells
        assert v_shifted.shape == (20, 20)
        assert v_shifted[0, 3] == v[0, 0]


class TestValidation:
    def test_valid_model(self):
        v = sm.gradient_background(20, 20, dtype="f32")
        sm.validate_model(v)  # Ne doit pas lever d'exception

    def test_invalid_dtype(self):
        v = sm.gradient_background(20, 20).astype(np.int32)
        with pytest.raises(ValueError, match="Model dtype must be float32 or float64"):
            sm.validate_model(v)
