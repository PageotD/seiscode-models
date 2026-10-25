"""Tests for geological models and precision behavior."""

import numpy as np
import pytest

import seiscode_models as sm


class TestGeologicalModels:
    @pytest.mark.parametrize(
        "model_func",
        [
            sm.dipping_layers,
            sm.sinusoidal_interface,
            sm.faulted_layers,
            sm.checkerboard,
            sm.salt_dome,
            sm.low_velocity_layer,
            sm.heterogeneous_anomaly_model,
        ],
    )
    def test_model_generation_and_dtype(self, model_func):
        # Test f32
        v_f32 = model_func(nx=50, nz=30, dtype="f32")
        assert v_f32.shape == (50, 30)
        assert v_f32.dtype == np.float32
        sm.validate_model(v_f32)

        # Test f64
        v_f64 = model_func(nx=50, nz=30, dtype="f64")
        assert v_f64.shape == (50, 30)
        assert v_f64.dtype == np.float64
        sm.validate_model(v_f64)

    def test_physical_parameters(self):
        # Test low_velocity_layer with z0 and thickness in meters
        v_lvl = sm.low_velocity_layer(
            nx=40, nz=40, dx=5.0, dz=2.0, z0=20.0, thickness=10.0, v_slow=1000.0
        )
        # z0=20m -> row 10, thickness=10m -> rows 10 to 15
        assert v_lvl[0, 10] == 1000.0
        assert v_lvl[0, 14] == 1000.0
        assert v_lvl[0, 5] != 1000.0

        # Test checkerboard with cell_x and cell_z
        v_cb = sm.checkerboard(nx=40, nz=40, dx=2.0, dz=2.0, cell_x=20.0, cell_z=10.0)
        assert v_cb.shape == (40, 40)
        sm.validate_model(v_cb, dx=2.0, dz=2.0)

    def test_layered_model(self):
        v = sm.layered_model(
            nx=30,
            nz=50,
            dx=2.0,
            dz=2.0,
            thicknesses=(20.0, 30.0),
            velocities=(1500.0, 2500.0, 3500.0),
            dtype="f32",
        )
        assert v.shape == (30, 50)
        assert v.dtype == np.float32
        # layer 1: 0 to 20m -> cells 0 to 10
        assert v[0, 0] == 1500.0
        assert v[0, 9] == 1500.0
        # layer 2: 20 to 50m -> cells 10 to 25
        assert v[0, 10] == 2500.0
        assert v[0, 24] == 2500.0
        # basement: 50m to 100m -> cells 25 to 50
        assert v[0, 25] == 3500.0
        assert v[0, 49] == 3500.0

        # Test VelocityModel.layered
        model = sm.VelocityModel.layered(
            nx=30,
            nz=50,
            dx=2.0,
            dz=2.0,
            thicknesses=(20.0, 30.0),
            velocities=(1500.0, 2500.0, 3500.0),
        )
        assert isinstance(model, sm.VelocityModel)
        assert np.array_equal(model.data, v)

    def test_layered_model_invalid_lengths(self):
        with pytest.raises(ValueError, match="Expected len"):
            sm.layered_model(thicknesses=(50.0,), velocities=(1500.0, 2000.0, 3000.0))

    def test_compacted_sediments(self):
        v = sm.compacted_sediments(nx=40, nz=30, dx=5.0, dz=10.0, v0=1500.0, k=0.5)
        assert v.shape == (40, 30)
        assert v[0, 0] == 1500.0
        # at row 10: depth = 10 * 10 = 100m -> 1500 + 0.5 * 100 = 1550 m/s
        assert v[0, 10] == 1550.0

        model = sm.VelocityModel.compacted(
            nx=40, nz=30, dx=5.0, dz=10.0, v0=1500.0, k=0.5
        )
        assert isinstance(model, sm.VelocityModel)
        assert np.array_equal(model.data, v)

    def test_add_water_bottom(self):
        bg = sm.gradient_background(50, 40, dx=5.0, dz=5.0, v0=2000.0, v1=4000.0)
        # Flat water depth = 25m (5 cells)
        marine = sm.add_water_bottom(
            bg, water_depth=25.0, v_water=1500.0, dx=5.0, dz=5.0
        )
        assert np.all(marine[:, :5] == 1500.0)
        assert np.all(marine[:, 5:] > 1500.0)

        # Profile bathymetry
        depths = np.linspace(10.0, 50.0, 50)
        marine_bathy = sm.add_water_bottom(
            bg, water_depth=depths, v_water=1480.0, dx=5.0, dz=5.0
        )
        assert marine_bathy[0, 0] == 1480.0
        assert marine_bathy[0, 3] > 1480.0

        # Method chaining on VelocityModel
        model = sm.VelocityModel.gradient(nx=50, nz=40, dx=5.0, dz=5.0).add_water_layer(
            water_depth=25.0
        )
        assert np.all(model.data[:, :5] == 1500.0)

    def test_dipping_layers_thickness_uniform(self):
        # 3 layers of constant thickness 20m at 0 dip
        v = sm.dipping_layers(
            nx=20,
            nz=50,
            dx=2.0,
            dz=2.0,
            v=(1000.0, 2000.0, 3000.0, 4000.0),
            dip_deg=0.0,
            thickness=20.0,
        )
        # 20m / 2m = 10 cells per layer
        assert v[0, 5] == 1000.0
        assert v[0, 15] == 2000.0
        assert v[0, 25] == 3000.0
        assert v[0, 35] == 4000.0

    def test_salt_dome_custom_apex_and_gradient(self):
        v = sm.salt_dome(
            nx=100,
            nz=60,
            dx=5.0,
            dz=5.0,
            cx=250.0,
            width=50.0,
            z_top=100.0,
            v_salt=4500.0,
            v_sed=2000.0,
            v_sed_bottom=3000.0,
        )
        assert v.shape == (100, 60)
        # Above apex z_top=100m (row < 20), no salt at cx=250m (col 50)
        assert v[50, 10] < 4500.0
        # Inside the dome below apex, salt is present
        assert v[50, 40] == 4500.0
