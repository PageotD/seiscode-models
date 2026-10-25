"""Visualization utilities compatible with Python 3.8+."""

from typing import Any, Optional, Tuple

import numpy as np

from .validation import validate_model


def quick_view(
    velocities: np.ndarray,
    dx: float = 1.0,
    dz: float = 1.0,
    title: str = "Velocity model",
    ax: Optional[Any] = None,
) -> Tuple[Any, Any]:
    """Display a 2D velocity field with depth positive downward using matplotlib.

    Renders the velocity model with correct physical coordinates (extent in meters),
    an automated aspect ratio, the viridis colormap, and a labeled colorbar.

    Parameters
    ----------
    velocities : ndarray of shape (nx, nz)
        2D velocity field to display, in m/s.
    dx : float, optional
        Grid spacing along horizontal x-axis in meters. Default is 1.0.
    dz : float, optional
        Grid spacing along vertical z-axis in meters. Default is 1.0.
    title : str, optional
        Title string displayed on the plot figure. Default is "Velocity model".
    ax : matplotlib.axes.Axes, optional
        Pre-existing Matplotlib Axes to draw the velocity model into. If None,
        a new figure and axes are created and `plt.show()` is called. Default is None.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        Figure and Axes containing the rendered model plot.

    Raises
    ------
    ValueError
        If `velocities` fails validation or if grid intervals are non-positive.
    """
    import matplotlib.pyplot as plt

    validate_model(velocities, dx=dx, dz=dz)

    show = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        show = True
    else:
        fig = ax.figure

    nx, nz = velocities.shape
    im = ax.imshow(
        velocities.T,
        extent=[0, (nx - 1) * dx, (nz - 1) * dz, 0],
        aspect="auto",
        cmap="viridis",
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Depth z (m)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Velocity (m/s)")
    if show:
        plt.show()
    return fig, ax
