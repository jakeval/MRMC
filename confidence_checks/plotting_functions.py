from typing import Tuple, Sequence, Optional
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


def plot_path(
    path: pd.DataFrame,
    path_color: str,
    marker: str = "o",
    path_label=None,
    alpha=None,
):
    """Plots a recourse path.

    Args:
        path: The path to plot.
        path_color: The color to use while plotting the path.
        path_index: The integer ID of the path to use as a label.
    """

    sns.scatterplot(x="x", y="y", data=path, color="grey", marker=marker)
    for i in range(path.shape[0] - 1):
        p1 = path.iloc[i]
        p2 = path.iloc[i + 1]
        label = None
        if i + 1 == path.shape[0] - 1 and path_label:
            label = path_label
        plt.plot(
            [p1.x, p2.x],
            [p1.y, p2.y],
            label=label,
            color=path_color,
            alpha=alpha,
        )


def plot_paths_results(
    paths_df: pd.DataFrame,
    index_df: pd.DataFrame = None,
    label_column: Optional[str] = None,
    run_colors: Optional[Sequence[str]] = None,
    path_markers: Optional[Sequence[str]] = None,
    alpha: Optional[float] = None,
):
    """Plots the paths of many runs overlapping on the same figure. Paths from
    the same run have the same color, but have different point markers.

    Args:
        paths_df: The paths to plot.
        index_df: The experiment results index dataframe.
        label_column: If the runs all vary only one parameter, each run will
            be labeled with its value for this parameter.
        run_colors: A sequence of colors to use when plotting runs one by one.
        path_markers: A sequence of markers to use when plotting a run's paths
            one by one.
        alpha: The alpha transparency to use when plotting paths."""
    run_colors = run_colors or [
        "green",
        "orange",
        "red",
        "royalblue",
        "peru",
        "lawngreen",
        "olive",
        "deeppink",
    ]
    path_markers = path_markers or ["o", "s", "P", "*", "D"]

    path_ids = paths_df.path_id.unique()
    run_ids = paths_df.run_id.unique()

    for run_id, run_color in zip(run_ids, run_colors):

        if not label_column:
            label = f"Run {run_id} paths"
        else:
            column_value = index_df[index_df.run_id == run_id][
                label_column
            ].iloc[0]
            label = f"Paths with {label_column}={column_value}"
        plt.plot(
            [-10, -10],
            [-12, -12],
            color=run_color,
            label=label,
        )[0]

        for path_id, path_marker in zip(path_ids, path_markers):
            path = paths_df[
                (paths_df.run_id == run_id) & (paths_df.path_id == path_id)
            ]
            plot_path(
                path,
                run_color,
                path_label=None,
                marker=path_marker,
                alpha=alpha,
            )

    for path_id, path_marker in zip(path_ids, path_markers):
        plt.scatter(
            [-10],
            [-10],
            marker=path_marker,
            color="grey",
            label=f"Recourse path {path_id}",
        )

    plt.xlim(-1.2, 1.3)
    plt.ylim(-0.2, 2.3)
    plt.legend()
