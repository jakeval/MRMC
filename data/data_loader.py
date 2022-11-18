import numpy as np
import pandas as pd
from typing import Optional, Mapping, Tuple
from data.datasets import credit_card_default_loader, base_loader
import enum


# TODO(@jakeval): Docstrings terminal quotations should be on new line.


class DatasetName(enum.Enum):
    """Enum class for dataset names.

    Dataset names are used to identify, load, and save datasets."""

    CREDIT_CARD_DEFAULT = credit_card_default_loader.DATASET_NAME


def load_data(
    dataset_name: DatasetName,
    data_dir: Optional[str] = None,
    loader_kwargs: Optional[Mapping] = None,
) -> Tuple[pd.DataFrame, base_loader.DatasetInfo]:
    """Returns the DataFrame and DatasetInfo class for the requested dataset.

    Args:
        dataset_name: The name to identify the dataset.
        data_dir: If provided, where to load data from (or download data to if
            not found). Defaults to MRMC/raw_data/.
        loader_kwargs: Optional key word arguments for the dataset-specific
            loader. For example, the CreditCardDefaultLoader accepts the
            only_continuous argument.

    Returns:
        A DataFrame containing the requested data and a DatasetInfo class
        describing the DataFrame columns."""
    loader_kwargs = loader_kwargs or {}
    if dataset_name == DatasetName.CREDIT_CARD_DEFAULT:
        loader = credit_card_default_loader.CreditCardDefaultLoader(
            data_dir=data_dir, **loader_kwargs
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} isn't supported.")
    return loader.load_data(), loader.dataset_info


def load_credit_card_default_dataset(
    only_continuous_vars: bool = True, data_dir: Optional[str] = None
) -> pd.DataFrame:
    """Returns the training data for the Credit Card Default dataset.

    Args:
        only_continuous_vars: Whether to drop or recategorize categorical
            variables, leaving only continuous features.
        data_dir: An override for the directory to load the data from (or
            download it to if unavailable).

    Returns:
        The Credit Card Default dataset."""
    return credit_card_default_loader.CreditCardDefaultLoader(
        only_continuous_vars=only_continuous_vars, data_dir=data_dir
    ).load_data()


# TODO(@jakeval): This will be removed in a later PR
def filter_from_poi(
    dataset, poi, immutable_features=None, feature_tolerances=None
):
    df = dataset[dataset.index != poi.index[0]]
    if immutable_features is not None:
        for feature in immutable_features:
            mask = None
            feature_value = poi.loc[poi.index[0], feature]
            if (
                feature_tolerances is not None
                and feature in feature_tolerances
            ):
                tol = feature_tolerances[feature]
                mask = np.abs(df[feature] - feature_value) <= tol
            else:
                mask = df[feature] == feature_value
            df = df.loc[mask, :]
    return df


# TODO(@jakeval): This will be removed in a later PR
def filter_from_model(dataset, model_scores, certainty_cutoff=0.7):
    Y_indices = (
        dataset.Y.mask(dataset.Y == -1, 0).to_numpy().astype("int32")
    )  # 0-1 representation
    correct_proba = np.where(
        Y_indices, model_scores[:, 1], model_scores[:, 0]
    )  # the probability of the correct label
    high_certainty_idx = correct_proba >= certainty_cutoff
    return dataset[high_certainty_idx]
