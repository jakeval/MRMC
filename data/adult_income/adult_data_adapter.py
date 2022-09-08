import pandas as pd
import numpy as np
from typing import Union, Tuple, Mapping, Sequence
import os
from data import data_preprocessor as dp
import pathlib


COLUMN_NAMES = 'age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income'.split(',')
TRAIN_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
TEST_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
TRAIN_FILENAME = 'adult.data'
TEST_FILENAME = 'adult.test'
RELATIVE_DATA_DIR = 'raw_data'
ABSOLUTE_DATA_DIR = pathlib.Path(__file__).parent / RELATIVE_DATA_DIR


def load_data(data_dir: Union[str, pathlib.Path] = ABSOLUTE_DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame, dp.Preprocessor]:
    """Loads the adult income dataset.
    
    If available locally as a csv at data_dir, loads it from there. Otherwise downloads it."""
    train_data, test_data = None, None
    if not os.path.exists(os.path.join(data_dir, TRAIN_FILENAME)) or not os.path.exists(os.path.join(data_dir, TEST_FILENAME)):
        train_data, test_data = download_data(data_dir)
    else:
        train_data, test_data = load_local_data(data_dir)

    data_df, test_df = process_data(train_data), process_data(test_data)
    
    category_features = ['workclass', 'occupation', 'race', 'sex', 'marital-status', 'education']
    continuous_features = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
    return data_df, test_df, dp.NaivePreprocessor(category_features, continuous_features, label='income').fit(data_df)


def download_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data = pd.read_csv(TRAIN_URL, sep=', ', names=COLUMN_NAMES)
    test_data = pd.read_csv(TEST_URL, sep=', ', names=COLUMN_NAMES, skiprows=1)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    train_data.to_csv(os.path.join(data_dir, TRAIN_FILENAME), header=True, index=False)
    test_data.to_csv(os.path.join(data_dir, TEST_FILENAME), header=True, index=False)
    return train_data, test_data


def load_local_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data = pd.read_csv(os.path.join(data_dir, TRAIN_FILENAME))
    test_data = pd.read_csv(os.path.join(data_dir, TEST_FILENAME))
    return train_data, test_data


def recategorize_feature(column: pd.Series, inverse_category_dict: Mapping[str, Sequence[str]]):
    """Returns a Series where some values are remapped to others.
    
    Given a dictionary like {'new_val': [val1, val2]}, val1 and val2 are relabeled as new_val in the new Series."""
    new_column = column.copy()
    for key, val_list in inverse_category_dict.items():
        for val in val_list:
            new_column = np.where(new_column == val, key, new_column)
    return new_column


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Processes the adult dataset.
    
    Drops three columns and simplifies marital status and education."""
    df = df.drop(columns=['fnlwgt', 'native-country', 'relationship', 'education-num'])
    df = df.drop_duplicates()

    marital_status_remapping = {
        'Single': ['Never-married', 'Divorced', 'Separated', 'Widowed'],
        'Married': ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse']
    }
    education_remapping = {
        'No-HS': ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th'],
        'HS-grad': ['HS-grad', 'Some-college'],
        'Bachelors': ['Bachelors'],
        'Graduate-program': ['Masters', 'Doctorate', 'Prof-school']
    }
    df['marital-status'] = recategorize_feature(df['marital-status'], marital_status_remapping)
    df['education'] = recategorize_feature(df['education'], education_remapping)
    df = df.drop(df[(df == '?').any(axis=1)].index)
    return df
