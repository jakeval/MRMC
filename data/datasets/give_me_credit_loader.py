from typing import Optional, Union, Sequence, Mapping
import pandas as pd
import numpy as np
from data.datasets import base_loader
from data.datasets import utils


DATASET_NAME = "give_me_credit"

_SPLIT_SEED = 12348571  # Random seed to use when creating train/val/test sets
_TRAIN_SPLIT_RATIO = 0.34
_VAL_SPLIT_RATIO = 0.33
_TEST_SPLIT_RATIO = 0.33

_SOURCE_TARGET_COLUMN = "SeriousDlqin2yrs"
_LABEL_COLUMN = "Y"

_DATASET_INFO = base_loader.DatasetInfo(
    continuous_features=[
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTimes30-59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTimes60-89DaysPastDueNotWorse",
        "NumberOfDependents",
    ],
    ordinal_features=[],
    categorical_features=[],
    label_column=_LABEL_COLUMN,
    positive_label=0,
    negative_label=1,
)


class GiveMeCreditLoader(base_loader.DataLoader):
    """Loads the Give Me Some Credit from local storage.

    Data is taken from
    https://www.kaggle.com/competitions/GiveMeSomeCredit/data."""

    def __init__(self, data_dir: Optional[str] = None, drop_na=True):
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
        self.drop_na = drop_na

    def download_data(self) -> pd.DataFrame:
        raise RuntimeError(
            "Downloading the Give Me Some Credit dataset is not supported."
        )

    def process_data(
        self,
        data: pd.DataFrame,
    ) -> Mapping[str, pd.DataFrame]:
        """Processes the Give Me Some Credit dataset and splits it into
        train/validation/test sets.

        Only the cs-train.csv file is used to create the splits because
        cs-test.csv does not have labels.

        Args:
            data: The data to process.

        Returns:
            The processed data split.
        """
        data = data.rename(
            columns={
                "Unnamed: 0": "index",
                _SOURCE_TARGET_COLUMN: _LABEL_COLUMN,
            }
        ).set_index("index")
        if self.drop_na:
            data = data[
                ~data.MonthlyIncome.isna() & ~data.NumberOfDependents.isna()
            ]
        return GiveMeCreditLoader._split_data(data)

    @staticmethod
    def _split_data(data: pd.DataFrame) -> Mapping[str, pd.DataFrame]:
        """Randomly splits a DataFrame into train, validation, and test
        partitions.

        The splits preserve the distribution of labels from the original
        dataset."""
        # split the data by label to ensure equal distribution
        pos_data = data[
            data[_DATASET_INFO.label_column] == _DATASET_INFO.positive_label
        ]
        neg_data = data[
            data[_DATASET_INFO.label_column] != _DATASET_INFO.positive_label
        ]

        # get the indices for each split per-label
        pos_split_indices = GiveMeCreditLoader._split_indices(
            pos_data.shape[0],
            _TRAIN_SPLIT_RATIO,
            _VAL_SPLIT_RATIO,
            _TEST_SPLIT_RATIO,
        )

        neg_split_indices = GiveMeCreditLoader._split_indices(
            neg_data.shape[0],
            _TRAIN_SPLIT_RATIO,
            _VAL_SPLIT_RATIO,
            _TEST_SPLIT_RATIO,
        )

        # construct the splits from the indices
        data_splits = {}
        for split in ["train", "val", "test"]:
            data_splits[split] = pd.concat(
                [
                    pos_data.iloc[pos_split_indices[split]],
                    neg_data.iloc[neg_split_indices[split]],
                ]
            )

        return data_splits

    @staticmethod
    def _split_indices(
        num_indices: int,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> Mapping[str, np.ndarray]:
        """Partitions a set of indices given the train and validation ratios.

        Returns a mapping from split name ('train', 'val', 'test') to the set
        of indices that should appear in that split.

        The dataset is assumed to be 0-indexed in increments of 1."""
        if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
            raise RuntimeError("Dataset split ratios must be non-negative.")
        if np.abs((train_ratio + val_ratio + test_ratio) - 1) > 1e-5:
            raise RuntimeError("Dataset split ratios must sum to 1.")
        rng = np.random.default_rng(seed=_SPLIT_SEED)
        indices = np.arange(num_indices)
        rng.shuffle(indices)
        train_index = int(np.floor(num_indices * train_ratio))
        val_index = int(np.floor(num_indices * (train_ratio + val_ratio)))
        return {
            "train": indices[:train_index],
            "val": indices[train_index:val_index],
            "test": indices[val_index:],
        }
