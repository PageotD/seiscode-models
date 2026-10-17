"""Visualization utilities compatible with Python 3.8+."""

import numpy as np

from .validation import validate_model


def quick_view(
    velocities: np.ndarray,
    dx: float = 1.0,
    dz: float = 1.0,
    title: str = "Velocity model",
) -> None:
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

    Raises
    ------
    ValueError
        If `velocities` fails validation or if grid intervals are non-positive.
    """
    import matplotlib.pyplot as plt

    validate_model(velocities, dx=dx, dz=dz)

    nx, nz = velocities.shape
    fig, ax = plt.subplots(figsize=(12, 6))
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
    plt.show()
