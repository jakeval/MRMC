import numpy as np
import pandas as pd
from data import recourse_adapter
from typing import Any


MIN_DIRECTION = 1e-32
"""Numbers smaller than this won't be used as the denominator during
division."""


def randomly_perturb_direction(
    direction: recourse_adapter.EmbeddedSeries, ratio: float
) -> recourse_adapter.EmbeddedSeries:
    """Randomly changes a vector's direction while maintaining its magnitude.

    The amount of random perturbation is determined by the `ratio` argument.
    If ratio=0.5, then random noise with magnitude 50% of the original
    direction is added. If ratio=1, then random noise with magnitude equal to
    the original direction is added. The direction is always rescaled to have
    its original magnitude after adding the random noise.

    Args:
        direction: The vector to perturb.
        ratio: The amount of random noise to add as a ratio of the direction's
            original magnitude.

    Returns:
        A new vector of equal magnitude to the original but with a randomly
        perturbed direction.
    """
    norm = np.linalg.norm(direction)
    noise = np.random.normal(0, 1, len(direction))
    noise = (noise / np.linalg.norm(noise)) * ratio * norm
    new_direction = direction + noise
    new_direction = (new_direction / np.linalg.norm(new_direction)) * norm
    return new_direction


def constant_step_size(
    direction: recourse_adapter.EmbeddedSeries, step_size: float = 1
) -> recourse_adapter.EmbeddedSeries:
    """Rescales a vector to a given fixed size measured by L2 norm.

    Args:
        direction: The vector to rescale.
        step_size: The target L2 norm of the rescaled vector.

    Returns:
        A new vector with direction equal to the original but rescaled to the
        magnitude given by `step_size`.
    """
    normalization = np.linalg.norm(direction.to_numpy())
    if normalization <= MIN_DIRECTION:
        return direction
    return (step_size * direction) / normalization


def random_poi(
    dataset: pd.DataFrame,
    label_column: str = "Y",
    label_value: Any = -1,
    drop_label: bool = True,
) -> pd.Series:
    """Selects a random POI of the given label from the dataset.

    Args:
        dataset: The dataset to sample from.
        label_column: The dataset column containing the class labels.
        label_value: The label value of the point to select.
        drop_label: Whether to drop the label from the returned POI.

    Returns:
        A random row of the given label from the dataset.
    """
    poi = dataset[dataset[label_column] == label_value].sample(1)
    if drop_label:
        poi = poi.drop(label_column, axis=1)

    return poi.iloc[0]
