from __future__ import annotations
from typing import Tuple, Sequence
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import decomposition

from data import recourse_adapter
from data.datasets import base_loader
from data.adapters import continuous_adapter
from models import model_interface


"""
plot_model_confidence()
 --model confidence contours

plot_direction()
 --pass

plot_path()
 --...

plot_paths_results()
 --...

"""


class PCAAdapter(recourse_adapter.RecourseAdapter):
    def __init__(self, label_column, positive_label):
        super().__init__(
            label_column=label_column,
            positive_label=positive_label,
        )
        self.standardizing_adapter = continuous_adapter.StandardizingAdapter(
            label_column=label_column, positive_label=positive_label
        )
        self.pca = None

    def fit(self, dataset: pd.DataFrame) -> PCAAdapter:
        super().fit(dataset)
        self.standardizing_adapter.fit(dataset)
        standardized_data = self.standardizing_adapter.transform(dataset)
        pca = decomposition.PCA(n_components=2)
        pca.fit(standardized_data.drop(self.label_column, axis=1))
        self.pca = pca
        return self

    def transform(
        self, dataset: pd.DataFrame
    ) -> recourse_adapter.EmbeddedDataFrame:
        df = super().transform(dataset)
        if self.label_column in dataset.columns:
            labels = df[self.label_column].to_numpy()
            df = df.drop(self.label_column, axis=1)
        standardized_df = self.standardizing_adapter.transform(df)
        X = self.pca.transform(standardized_df)
        pca_df = pd.DataFrame(data=X, columns=["pca_1", "pca_2"])
        if self.label_column in dataset.columns:
            pca_df[self.label_column] = labels
        return pca_df

    def inverse_transform(
        self, dataset: pd.DataFrame
    ) -> recourse_adapter.EmbeddedDataFrame:
        df = super().inverse_transform(dataset)
        if self.label_column in df.columns:
            labels = df[self.label_column].to_numpy()
            df = df.drop(self.label_column, axis=1)
        X = self.pca.inverse_transform(df)
        df = pd.DataFrame(data=X, columns=self.column_names())
        df = self.standardizing_adapter.inverse_transform(df)
        if self.label_column in df.columns:
            df[self.label_column] = labels
        return df

    def column_names(self, drop_label=True):
        return self.standardizing_adapter.columns.difference(
            [self.label_column]
        )

    def embedded_column_names(self, drop_label=True):
        pass

    def directions_to_instructions(self, directions):
        pass

    def interpret_instructions(self, poi, instructions):
        pass


def plot_stuff(
    model: model_interface.Model,
    dataset: pd.DataFrame,
    dataset_info: base_loader.DatasetInfo,
    paths: Sequence[pd.DataFrame],
    confidence_cutoff: float,
    xlim: Tuple(float, float) = None,
    ylim: Tuple(float, float) = None,
):
    adapter = PCAAdapter(
        label_column=dataset_info.label_column,
        positive_label=dataset_info.positive_label,
    ).fit(dataset)
    pca_data = adapter.transform(dataset)
    xy_data = pca_data.drop(dataset_info.label_column, axis=1)
    buffer = 0.2
    maxval = xy_data.max().max() + buffer
    minval = xy_data.min().min() - buffer
    axis = np.linspace(minval, maxval, 50)
    x_grid, y_grid = np.meshgrid(axis, axis)
    grid_dataset = pd.DataFrame(
        {"pca_1": x_grid.flatten(), "pca_2": y_grid.flatten()}
    )
    scores = model.predict_pos_proba(
        adapter.inverse_transform(grid_dataset)
    ).to_numpy()
    z_grid = np.reshape(scores, x_grid.shape)

    plt.contourf(x_grid, y_grid, z_grid, levels=np.linspace(0, 1, 11))
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("Model Positive Probability")

    sns.scatterplot(
        data=pca_data,
        x="pca_1",
        y="pca_2",
        hue=dataset_info.label_column,
        alpha=0.7,
    )

    for path in paths:
        pca_path = adapter.transform(path)
        plot_path(
            path=pca_path.rename(columns={"pca_1": "x", "pca_2": "y"}),
            path_color="green",
        )
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)


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
    paths_df,
    index_df=None,
    label_column=None,
    run_colors=None,
    path_markers=None,
    alpha=None,
):
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
