import numpy as np
import pandas as pd
from data import recourse_adapter
from typing import Any


MIN_DIRECTION = 1e-32


def randomly_perturb_direction(
    direction: recourse_adapter.EmbeddedSeries, ratio: float
):
    norm = np.linalg.norm(direction)
    noise = np.random.normal(0, 1, len(direction))
    noise = (noise / np.linalg.norm(noise)) * ratio * norm
    new_direction = direction + noise
    new_direction = (new_direction / np.linalg.norm(new_direction)) * norm
    return new_direction


def constant_step_size(direction, step_size=1):
    normalization = np.linalg.norm(direction.to_numpy())
    if normalization <= MIN_DIRECTION:
        return direction
    return (step_size * direction) / normalization


def random_poi(
    dataset: pd.DataFrame, column="Y", label: Any = -1, drop_label: bool = True
) -> pd.Series:
    """Selects a random POI of the given label from the dataset.

    Args:
        dataset: The dataset to sample from.
        label: The label of the point to select.
        drop_label: Whether to drop the label from the returned POI.
    """
    poi = dataset[dataset[column] == label].sample(1)
    if drop_label:
        poi = poi.drop(column, axis=1)

    return poi.iloc[0]
