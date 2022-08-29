import numpy as np
import pandas as pd
from typing import Tuple
from data import adult_data_adapter as adult_da
from data import german_data_adapter as german_da
from data import synthetic_data_adapter as synthetic_da
from data import data_preprocessor as dp


def load_adult_income_dataset(data_dir: str = '../data/adult') -> Tuple[pd.DataFrame, pd.DataFrame, dp.Preprocessor]:
    """Returns training data, test data, and a preprocessor."""
    return adult_da.load_data(data_dir)


def load_german_credit_dataset(data_dir='../data/german') -> Tuple[pd.DataFrame, pd.DataFrame, dp.Preprocessor]:
    """Returns training data, test data, and a preprocessor."""
    return german_da.load_data(data_dir)


def load_synthetic_dataset() -> Tuple[pd.DataFrame, dp.Preprocessor]:
    """Returns a dataset and a preprocessor."""
    return synthetic_da.load_data()


def filter_from_poi(dataset, poi, immutable_features=None, feature_tolerances=None):
    df = dataset[dataset.index != poi.index[0]]
    if immutable_features is not None:
        for feature in immutable_features:
            mask = None
            feature_value = poi.loc[poi.index[0],feature]
            if feature_tolerances is not None and feature in feature_tolerances:
                tol = feature_tolerances[feature]
                mask = np.abs(df[feature] - feature_value) <= tol
            else:
                mask = df[feature] == feature_value
            df = df.loc[mask, :]
    return df


def filter_from_model(dataset, model_scores, certainty_cutoff=0.7):
    Y_indices = dataset.Y.mask(dataset.Y == -1, 0).to_numpy().astype('int32') # 0-1 representation
    correct_proba = np.where(Y_indices, model_scores[:,1], model_scores[:,0]) # the probability of the correct label
    high_certainty_idx = correct_proba >= certainty_cutoff
    return dataset[high_certainty_idx]
