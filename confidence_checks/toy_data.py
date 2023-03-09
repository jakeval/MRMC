from typing import Tuple, Union, Sequence
import pandas as pd
import numpy as np

from data.datasets import base_loader

_RANDOM_SEED = 91357

_DATASET_INFO = base_loader.DatasetInfo(
    continuous_features=["x", "y"],
    ordinal_features=[],
    categorical_features=[],
    label_column="label",
    positive_label=1,
    negative_label=-1,
)


def get_data(
    split: Union[str, Sequence[str]] = "train",
) -> Tuple[Union[pd.DataFrame, base_loader.DatasetInfo], ...]:
    """Generates a simple synthetic dataset:

    * 1 negatively-classified POI at xy-coordinates (-1, 0).
    * 1 positively-classified 4-point cluster at xy-coords (0, 2).
    * 1 positively-classified 4-point cluster at xy-coords (1, 2).

    Returns:
        A DataFrame and DatasetInfo object.
    """
    train_data = _get_train()

    data = {
        "train": train_data,
        "val": _get_eval(train_data),
        "test": _get_eval(train_data),
    }

    if type(split) == str:
        split = [split]

    return *[data[split_name] for split_name in split], _DATASET_INFO


def _get_train() -> pd.DataFrame:
    offsets = np.array([[-0.1, -0.1], [-0.1, 0.1], [0.1, -0.1], [0.1, 0.1]])
    grid_1 = np.array([[2, 1]] * 4) + offsets
    grid_2 = np.array([[2, 0]] * 4) + offsets
    pois = np.array([[0.1, -0.9]] * 4) + offsets
    data = np.concatenate([grid_1, grid_2, pois])
    labels = [1] * 8 + [-1] * 4
    return pd.DataFrame({"x": data[:, 1], "y": data[:, 0], "label": labels})


def _get_eval(train: pd.DataFrame) -> pd.DataFrame:
    eval_data = train.copy()
    rng = np.random.RandomState(_RANDOM_SEED)
    noise = rng.normal(0, 0.05, size=(eval_data.shape[0], 2))
    eval_data.loc[:, ["x", "y"]] = eval_data.loc[:, ["x", "y"]] + noise
    return eval_data
