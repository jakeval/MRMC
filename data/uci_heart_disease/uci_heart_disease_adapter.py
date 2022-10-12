import pandas as pd
import numpy as np
from typing import Union, Tuple, Mapping, Sequence
import os
from data import data_preprocessor as dp
import pathlib

COLUMN_NAMES = 'age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,num'.split(',')
URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
DATA_FILENAME = 'processed.cleveland.data'
RELATIVE_DATA_DIR = 'raw_data'
ABSOLUTE_DATA_DIR = pathlib.Path(__file__).parent / RELATIVE_DATA_DIR


def load_data(data_dir: Union[str, pathlib.Path] = ABSOLUTE_DATA_DIR) -> Tuple[pd.DataFrame, dp.Preprocessor]:
    """Loads the UCI heart disease dataset.
    
    If available locally as a csv at data_dir, loads it from there. Otherwise downloads it."""
    data = None
    if not os.path.exists(os.path.join(data_dir, DATA_FILENAME)):
        data = download_data(data_dir)
    else:
        data = load_local_data(data_dir)

    data_df = process_data(data)

    continuous_features = data_df.columns
    return data_df, dp.NaivePreprocessor([], continuous_features, label='num').fit(data_df)


def download_data(data_dir: str) -> pd.DataFrame:
    data = pd.read_csv(URL, sep=',', names=COLUMN_NAMES)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    data.to_csv(os.path.join(data_dir, DATA_FILENAME), header=True, index=False)
    return data


def load_local_data(data_dir: str) -> pd.DataFrame:
    data = pd.read_csv(os.path.join(data_dir, DATA_FILENAME))
    return data


def recategorize_feature(column: pd.Series, inverse_category_dict: Mapping[str, Sequence[str]]):
    """Returns a Series where some values are remapped to others.

    Given a dictionary like {'new_val': [val1, val2]}, val1 and val2 are relabeled as new_val in the new Series."""
    new_column = column.copy()
    for key, val_list in inverse_category_dict.items():
        for val in val_list:
            new_column[column == val] = key
    return new_column


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Processes the adult dataset.
    
    Drops three columns and simplifies marital status and education."""
    df = df.drop(columns=['cp'])
    mask = (df != '?').all(axis=1)
    df = df.loc[mask]

    df = df.astype('float64')

    restecg_remapping = {
        0: [0],
        1: [1,2]
    }
    thal_remapping = {
        0: [3],
        1: [6,7]
    }
    num_remapping = {
        1: [0],
        0: [1,2,3,4]
    }
    df['restecg'] = recategorize_feature(df['restecg'], restecg_remapping)
    df['thal'] = recategorize_feature(df['thal'], thal_remapping)
    df['num'] = recategorize_feature(df['num'], num_remapping)

    return df
