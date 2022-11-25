from typing import Tuple
import pandas as pd
import numpy as np

from data.datasets import base_loader


_DATASET_INFO = base_loader.DatasetInfo(
    continuous_features=["x", "y"],
    ordinal_features=[],
    categorical_features=[],
    label_column="label",
    positive_label=1,
)


def get_data() -> Tuple[pd.DataFrame, base_loader.DatasetInfo]:
    """Generates a simple synthetic dataset:

    * 1 negatively-classified POI at (0, 0).
    * 1 positively-classified 4-point cluster at (1, 2).
    * 1 positively-classified 4-point cluster at (-1, 2).

    Returns:
        A DataFrame and DatasetInfo object.
    """
    offsets = np.array([[-0.1, -0.1], [-0.1, 0.1], [0.1, -0.1], [0.1, 0.1]])
    grid_1 = np.array([[2, 1]] * 4) + offsets
    grid_2 = np.array([[2, -1]] * 4) + offsets
    POI = np.array([[0, 0]])
    data = np.concatenate([grid_1, grid_2, POI])
    labels = [1] * 8 + [-1]
    return (
        pd.DataFrame({"x": data[:, 1], "y": data[:, 0], "label": labels}),
        _DATASET_INFO,
    )
