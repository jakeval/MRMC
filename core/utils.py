import numpy as np
import pandas as pd
from data import recourse_adapter
from typing import Any, Optional
from models import model_interface


_MIN_DIRECTION = 1e-32
"""Numbers smaller than this won't be used as the denominator during
division."""


def randomly_perturb_direction(
    direction: recourse_adapter.EmbeddedSeries,
    ratio: float,
    random_generator: Optional[np.random.Generator] = None,
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
        random_generator: An optional random generator to use when perturbing
            the direction. Otherwise defaults to np.random.normal().

    Returns:
        A new vector of equal magnitude to the original but with a randomly
        perturbed direction.
    """
    # Check for zeroes to avoid division by zero.
    if ratio == 0:
        return direction
    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0:
        return direction

    if random_generator:
        noise = random_generator.normal(0, 1, len(direction))
    else:
        noise = np.random.normal(0, 1, len(direction))
    noise_norm = np.linalg.norm(noise)
    if noise_norm == 0:
        return direction

    noise = (noise / noise_norm) * ratio * direction_norm
    new_direction = direction + noise
    # Normalize noised direction and rescale to the original direction length.
    new_direction = (
        new_direction / np.linalg.norm(new_direction)
    ) * direction_norm
    return new_direction


# TODO(@jakeval): Revisit "step size" naming convention.
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
    normalization = np.linalg.norm(direction)
    if normalization == 0:
        return direction
    if normalization <= _MIN_DIRECTION:
        normalization = _MIN_DIRECTION
    return (step_size * direction) / normalization


# TODO(@jakeval): Add unit test for this (second pass).
def random_poi(
    dataset: pd.DataFrame,
    label_column: str,
    label_value: Any,
    model: model_interface.Model,
    drop_label: bool = True,
    random_seed: Optional[int] = None,
) -> pd.Series:
    """Selects a random POI of the given model classification from the dataset.

    Args:
        dataset: The dataset to sample from.
        label_column: The dataset column containing the class labels.
        label_value: The classification label of the point to select.
        model: The classification model to use when selecting a point.
        drop_label: Whether to drop the label from the returned POI.
        random_seed: An optional random seed to select the POI with.

    Returns:
        A random row of the given label from the dataset.
    """
    pred_labels = model.predict(dataset)
    poi = dataset[pred_labels == label_value].sample(
        1, random_state=random_seed
    )
    if drop_label:
        poi = poi.drop(label_column, axis=1)

    return poi.iloc[0]
