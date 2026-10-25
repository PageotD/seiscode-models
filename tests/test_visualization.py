"""Tests for visualization utilities."""

from unittest.mock import patch

import numpy as np
import pytest

import seiscode_models as sm


class TestQuickView:
    @patch("matplotlib.pyplot.show")
    def test_quick_view_execution(self, mock_show):
        v = sm.gradient_background(30, 20)
        # Ne doit pas lever d'exception et doit appeler matplotlib.pyplot.show
        fig, ax = sm.quick_view(v, dx=1.5, dz=1.5, title="Test Model")
        mock_show.assert_called_once()
        assert fig is not None
        assert ax is not None

    def test_quick_view_with_custom_ax(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        v = sm.gradient_background(30, 20)
        ret_fig, ret_ax = sm.quick_view(v, ax=ax)
        assert ret_fig is fig
        assert ret_ax is ax
        plt.close(fig)

    def test_velocity_model_plot(self):
        import matplotlib.pyplot as plt

        m = sm.VelocityModel.gradient(nx=30, nz=20)
        fig, ax = plt.subplots()
        ret_fig, ret_ax = m.plot(title="Custom Title", ax=ax)
        assert ret_fig is fig
        assert ret_ax is ax
        plt.close(fig)

    def test_quick_view_invalid_model(self):
        v = np.array([1, 2, 3])  # 1D invalide
        with pytest.raises(ValueError, match="Model must be 2D"):
            sm.quick_view(v)
