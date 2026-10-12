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
        sm.quick_view(v, dx=1.5, dz=1.5, title="Test Model")
        mock_show.assert_called_once()

    def test_quick_view_invalid_model(self):
        v = np.array([1, 2, 3]) # 1D invalide
        with pytest.raises(ValueError, match="Model must be 2D"):
            sm.quick_view(v)