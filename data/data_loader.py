import numpy as np
import pandas as pd
from typing import Optional
from data.datasets import credit_card_default_loader


def load_credit_card_default_dataset(
    only_continuous: bool = True, data_dir: Optional[str] = None
) -> pd.DataFrame:
    """Returns the training data for the Credit Card Default dataset.

    Args:
        only_continuous: Whether to drop or recategorize categorical variables,
            leaving only continuous features.
        data_dir: An override for the directory to load the data from (or
            download it to if unavailable).

    Returns:
        The Credit Card Default dataset."""
    return credit_card_default_loader.CreditCardDefaultLoader(
        only_continuous=only_continuous, data_dir=data_dir
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
