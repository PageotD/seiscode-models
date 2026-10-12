"""Tests for the object-oriented VelocityModel interface."""
import numpy as np
import pytest
import seiscode_models as sm

class TestVelocityModelOOP:
    def test_factory_and_properties(self):
        model = sm.VelocityModel.gradient(nx=100, nz=50, dx=2.0, dz=2.0, dtype="f32")
        assert model.nx == 100
        assert model.nz == 50
        assert model.dx == 2.0
        assert model.dz == 2.0
        assert model.dtype == np.float32
        assert model.data.shape == (100, 50)

    def test_method_chaining(self):
        model = (
            sm.VelocityModel.gradient(nx=80, nz=40, dtype="f64")
            .add_anomaly(cx=40, cz=20, sigma_x=5.0, sigma_z=5.0, dv=500.0)
            .add_fault(ix_fault=50, throw=5)
        )
        assert isinstance(model, sm.VelocityModel)
        assert model.dtype == np.float64
        assert model.data.shape == (80, 40)

    def test_export_from_model(self, tmp_path):
        model = sm.VelocityModel.salt_dome(30, 30)
        file_path = tmp_path / "oop_model.bin"
        model.export(file_path, format="bin")
        assert file_path.exists()