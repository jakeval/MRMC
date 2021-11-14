import numpy as np
import pandas as pd
from data import adult_data_adapter as adult_da


def load_adult_income_dataset(data_dir='../data/adult'):
    return adult_da.load_data(data_dir)


def random_poi(dataset, label=-1, drop_label=True):
    poi = dataset[dataset.Y == label].sample(1)
    if drop_label:
        poi = poi.drop("Y", axis=1)
    return poi


def filter_from_poi(dataset, poi, immutable_features=None, feature_tolerances=None):
    df = dataset.drop(poi.index, axis=0)
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
  