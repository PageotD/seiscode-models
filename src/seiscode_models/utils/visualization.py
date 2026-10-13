"""Visualization utilities compatible with Python 3.8+."""

import numpy as np

from .validation import validate_model


def quick_view(
    velocities: np.ndarray,
    dx: float = 1.0,
    dz: float = 1.0,
    title: str = "Velocity model",
) -> None:
    """Display a velocity field with depth positive downward.

    Parameters
    ----------
    velocities : ndarray (nx, nz)
        Velocity field to display, in m/s.
    dx : float, optional
        Grid spacing along x, in meters. Default is 1.0.
    dz : float, optional
        Grid spacing along z, in meters. Default is 1.0.
    title : str, optional
        Figure title. Default is "Velocity model".
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
