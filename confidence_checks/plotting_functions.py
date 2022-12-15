from typing import Tuple
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from models import model_interface


def plot_model_confidence(
    model: model_interface.Model,
    xlim: Tuple[float, float] = (-1, 1),
    ylim: Tuple[float, float] = (0, 2),
):
    """Plots model confidence contours.

    Args:
        model: The model whose decision boundary to plot as contour lines.
        xlim: The x-axis boundaries of the confidence gradient.
        ylim: The y-axis boundaries of the confidence gradient.
    """
    # Get coordinates of a data grid
    buffer = 0.2
    x_axis = np.linspace(xlim[0] - buffer, xlim[1] + buffer, 50)
    y_axis = np.linspace(ylim[0] - buffer, ylim[1] + buffer, 50)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)

    # Convert to a standard format dataset
    grid_dataset = pd.DataFrame({"x": x_grid.flatten(), "y": y_grid.flatten()})
    scores = model.predict_pos_proba(grid_dataset).to_numpy()
    z_grid = np.reshape(scores, x_grid.shape)

    # Plot the boundary
    plt.contourf(x_grid, y_grid, z_grid, levels=np.linspace(0, 1, 11))
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("Model Positive Probability")


def plot_direction(
    direction: pd.Series,
    poi: pd.Series,
    direction_color: str,
    direction_index: int,
):
    """Plots a recourse direction.

    Args:
        direction: The recourse direction to plot.
        poi: The origin of the recourse direction.
        direction_color: The color to use while plotting the line.
        direction_index: The integer ID of the direction to use as a label.
    """
    plt.plot(
        [poi.x, poi.x + direction.x],
        [poi.y, poi.y + direction.y],
        label=f"Recourse direction {direction_index}",
        color=direction_color,
    )
    plt.legend()


def plot_path(path: pd.DataFrame, path_color: str, path_index: int):
    """Plots a recourse path.

    Args:
        path: The path to plot.
        path_color: The color to use while plotting the path.
        path_index: The integer ID of the path to use as a label.
    """
    sns.scatterplot(x="x", y="y", data=path, color="grey")
    for i in range(path.shape[0] - 1):
        p1 = path.iloc[i]
        p2 = path.iloc[i + 1]
        label = None
        if i + 1 == path.shape[0] - 1:
            label = f"Recourse path {path_index}"
        plt.plot([p1.x, p2.x], [p1.y, p2.y], label=label, color=path_color)
