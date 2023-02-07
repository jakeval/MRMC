from typing import Optional
import pandas as pd
from data.datasets import base_loader


def _flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]


KEYPOINT_BODY_PARTS = [
    "eye",
    "shoulder",
    "elbow",
    "wrist",
    "hip",
    "knee",
    "ankle",
]


LEFT_RIGHT_KEYPOINTS = _flatten(
    [
        [f"left_{body_part}", f"right_{body_part}"]
        for body_part in KEYPOINT_BODY_PARTS
    ]
)


KEYPOINT_COLUMNS = _flatten(
    [[f"{keypoint}_x", f"{keypoint}_y"] for keypoint in LEFT_RIGHT_KEYPOINTS]
)


DATASET_NAME = "infinite_form"
_DATASET_INFO = base_loader.DatasetInfo(
    continuous_features=KEYPOINT_COLUMNS,
    ordinal_features=[],
    categorical_features=[],
    label_column="pose_category",
    positive_label="downdog",
)


class InfiniteFormLoader(base_loader.DataLoader):
    """Loads InfiniteForm Dataset. Only local loading is supported.

    Data is taken from
    https://pixelate.ai/InfiniteForm"""

    def __init__(self, data_dir: Optional[str] = None):
        """Creates a data loader for the UCI Credit Card Default dataset.

        Args:
            only_continuous_vars: If True, drops the categorical SEX column and
                             turns the categorical PAY_* columns into
                             continuous columns.
        """
        super().__init__(
            dataset_info=_DATASET_INFO,
            data_dir=data_dir,
            dataset_name=DATASET_NAME,
        )

    def download_data(self) -> pd.DataFrame:
        raise NotImplementedError(
            "Infinite Form can't be automatically downloaded"
        )

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.set_index("id")
