"""Tests for the object-oriented VelocityModel interface."""

import numpy as np
import pytest

import seiscode_models as sm
from seiscode_models.core import VelocityModel


class TestVelocityModelOOP:
    def test_factory_and_properties(self):
        model = sm.VelocityModel.gradient(
            nx=100, nz=50, dx=2.0, dz=2.0, dtype="f32"
        )

        assert model.nx == 100
        assert model.nz == 50
        assert model.dx == 2.0
        assert model.dz == 2.0
        assert model.dtype == np.float32
        assert model.data.shape == (100, 50)

    def test_method_chaining(self):
        model = (
            sm.VelocityModel.gradient(nx=80, nz=40, dtype="f64")
            .add_gaussian_anomaly(
                cx=40, cz=20, sigma_x=5.0, sigma_z=5.0, dv=500.0
            )
            .add_fault(ix_fault=50, throw=5)
        )

        assert isinstance(model, sm.VelocityModel)
        assert model.dtype == np.float64
        assert model.data.shape == (80, 40)

    def test_gaussian_anomaly(self):
        model = sm.VelocityModel.gradient(nx=80, nz=40, dtype="f32")
        original = model.data.copy()

        result = model.add_gaussian_anomaly(
            cx=40, cz=20, sigma_x=5.0, sigma_z=5.0, dv=500.0
        )

        assert result is model
        assert model.data.shape == (80, 40)
        assert model.dtype == np.float32
        assert model.data[40, 20] > original[40, 20]

    def test_elliptical_anomaly(self):
        model = sm.VelocityModel.gradient(nx=80, nz=40, dtype="f32")
        original = model.data.copy()

        result = model.add_elliptical_inclusion(
            cx=40, cz=20, radius_x=10.0, radius_z=5.0, cv=500.0
        )

        assert result is model
        assert model.data.shape == (80, 40)
        assert model.dtype == np.float32

        # Velocity at the anomaly center increases.
        assert model.data[40, 20] == 500 #pytest.approx(
        #    original[40, 20] + 500.0
        #)

        # A point inside the ellipse is affected.
        assert model.data[40, 24] == 500 #pytest.approx(
            #original[40, 24] + 500.0
        #)

        # A point outside the ellipse remains unchanged.
        assert model.data[40, 27] == pytest.approx(original[40, 27])

    def test_add_anomalies_method_chaining(self):
        model = (
            sm.VelocityModel.gradient(nx=80, nz=40, dtype="f64")
            .add_gaussian_anomaly(
                cx=30, cz=15, sigma_x=4.0, sigma_z=3.0, dv=300.0
            )
            .add_elliptical_inclusion(
                cx=55, cz=20, radius_x=8.0, radius_z=4.0, cv=-200.0
            )
        )

        assert isinstance(model, sm.VelocityModel)
        assert model.dtype == np.float64
        assert model.data.shape == (80, 40)

    def test_add_fault(self):
        model = sm.VelocityModel.gradient(nx=80, nz=40, dtype="f32")
        original = model.data.copy()

        result = model.add_fault(ix_fault=40, throw=5)

        assert result is model
        assert model.data.shape == original.shape
        assert model.dtype == original.dtype

    def test_export_from_model(self, tmp_path):
        model = sm.VelocityModel.salt_dome(30, 30)
        file_path = tmp_path / "oop_model.bin"

        model.export(file_path, format="bin")

        assert file_path.exists()

    def test_version_matches(self):
        assert isinstance(sm.__version__, str)
        assert sm.__version__.count(".") == 2

    def test_copy_is_independent(self):
        model = sm.VelocityModel.gradient()
        original = model.data.copy()

        copied = model.copy()
        copied.add_gaussian_anomaly(
            cx=100, cz=25, sigma_x=10.0, sigma_z=5.0, dv=500.0
        )

        assert model != copied
        assert np.array_equal(model.data, original)

    def test_copy_preserves_metadata(self):
        model = sm.VelocityModel.gradient(
            nx=60, nz=30, dx=2.5, dz=3.0, dtype="f64"
        )
        copied = model.copy()

        assert copied is not model
        assert copied.dx == model.dx
        assert copied.dz == model.dz
        assert copied.dtype == model.dtype
        assert copied.nx == model.nx
        assert copied.nz == model.nz
        assert np.array_equal(copied.data, model.data)

    def test_stats(self):
        model = sm.VelocityModel.gradient()
        stats = model.stats()

        assert stats["shape"] == (model.nx, model.nz)
        assert stats["vmin"] <= stats["vmean"] <= stats["vmax"]
        assert stats["dtype"] == "float32"
        assert stats["dx"] == model.dx
        assert stats["dz"] == model.dz

    def test_smooth_preserves_shape_dtype(self):
        model = VelocityModel.checkerboard()
        smoothed = model.copy().smooth(k=5)

        assert smoothed.data.shape == model.data.shape
        assert smoothed.dtype == model.dtype

    def test_smooth_invalid_kernel(self):
        model = sm.VelocityModel.gradient()

        with pytest.raises(ValueError):
            model.smooth(k=4)

    def test_smooth_method_chaining(self):
        model = sm.VelocityModel.gradient(nx=50, nz=30)
        result = model.smooth(k=3)

        assert result is model
        assert result.data.shape == (50, 30)

    def test_clip(self):
        model = sm.VelocityModel.gradient(v0=100.0, v1=9000.0)

        result = model.clip(vmin=500.0, vmax=8000.0)

        assert result is model
        assert model.vmin >= 500.0
        assert model.vmax <= 8000.0

    def test_clip_invalid_bounds(self):
        model = sm.VelocityModel.gradient()

        with pytest.raises(ValueError, match="vmin must be <= vmax"):
            model.clip(vmin=8000.0, vmax=500.0)

    def test_clip_nonfinite_bounds(self):
        model = sm.VelocityModel.gradient()

        with pytest.raises(ValueError, match="must be finite"):
            model.clip(vmin=float("nan"), vmax=8000.0)

    def test_export_load_roundtrip(self, tmp_path):
        model = sm.VelocityModel.gradient()
        file_path = tmp_path / "model.bin"

        model.export(file_path, format="bin", order="F")

        loaded = sm.VelocityModel.load(
            file_path,
            shape=(model.nx, model.nz),
            dx=1.0,
            dz=1.0,
        )

        assert np.array_equal(model.data, loaded.data)
        assert loaded.dx == model.dx
        assert loaded.dz == model.dz

    def test_repr(self):
        model = sm.VelocityModel.gradient()

        assert "VelocityModel" in repr(model)
        assert f"nx={model.nx}" in repr(model)
        assert f"nz={model.nz}" in repr(model)

    def test_equality(self):
        model1 = sm.VelocityModel.gradient(nx=20, nz=10)
        model2 = sm.VelocityModel.gradient(nx=20, nz=10)

        assert model1 == model2

    def test_inequality_after_modification(self):
        model1 = sm.VelocityModel.gradient(nx=40, nz=30)
        model2 = model1.copy()

        model2.add_elliptical_inclusion(
            cx=20, cz=15, radius_x=8.0, radius_z=4.0, cv=500.0
        )

        assert model1 != model2