"""This file loads and processes the UCI credit card default dataset.

Dataset Features:
    LIMIT_BAL
        amount of the given credit. continuous
    SEX
        1/2 binary. 1=Male, 2=Female
    EDUCATION
        categorical
        1=grad
        2=university
        3=high school
        4=others

        The data also has values 0, 5, and 6, which are undocumented.
        The EDUCATION column is dropped because of these values.
    MARRIAGE
        categorical
        1=married
        2=single
        3=others

        The data also includes avalue 0 which is undocumented.
        The MARRIAGE column is dropped because of this value.
    AGE
        continuous
    PAY_*
        -2=paid in full and no recent transactions
        -1=paid in full but has positive balance due to recent transactions
        0=paid minimum amount
        1=payment delay for one month
        2=payment delay for 2 months
        ...
        9=payment delay for 9 or more months
        https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset/discussion/34608
    BILL_AMT*
        continuous
    PAY_AMT*
        continuous
"""


import pandas as pd
import numpy as np
from typing import Union, Tuple
import os
from data import data_preprocessor as dp
import pathlib
from core import utils


# TODO(@jakeval): The dataset adapter classes all look similar. Can they be abstracted?
TARGET_COLUMN = 'default payment next month'
URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls'
DATA_FILENAME = 'default_of_credit_card_clients.csv'
RELATIVE_DATA_DIR = 'raw_data'
ABSOLUTE_DATA_DIR = pathlib.Path(__file__).parent / RELATIVE_DATA_DIR


def load_data(only_continuous: bool = True, data_dir: Union[str, pathlib.Path] = ABSOLUTE_DATA_DIR) -> Tuple[pd.DataFrame, dp.Preprocessor]:
    """Loads the UCI Credit Card Default dataset.

    If available locally as a csv at data_dir, loads it from there. Otherwise downloads it."""
    data = None
    if not os.path.exists(os.path.join(data_dir, DATA_FILENAME)):
        data = download_data(data_dir)
    else:
        data = load_local_data(data_dir)

    data_df = process_data(data, only_continuous)

    continuous_features = data_df.columns
    return data_df, dp.NaivePreprocessor([], continuous_features, label='Y').fit(data_df)


def download_data(data_dir: str) -> pd.DataFrame:
    data = pd.read_excel(URL, header=1)

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    data.to_csv(os.path.join(data_dir, DATA_FILENAME), header=True, index=False)
    return data


def load_local_data(data_dir: str) -> pd.DataFrame:
    data = pd.read_csv(os.path.join(data_dir, DATA_FILENAME))
    return data


def process_data(df: pd.DataFrame, only_continuous: bool = True) -> pd.DataFrame:
    """Processes the adult dataset.

    Drops marriage and education columns. Renames PAY_0 and the label column. Groups PAY_* = 0, -1, -2 into one category."""
    df = df.set_index('ID').drop(columns=['MARRIAGE','EDUCATION']).rename(columns={TARGET_COLUMN: 'Y', 'PAY_0': 'PAY_1'})
    if only_continuous:
        pay_mapping = {
            0: [-1, -2]
        }
        for pay_column in [f'PAY_{i}' for i in range(1, 7)]:
            df[pay_column] = utils.recategorize_feature(df[pay_column], pay_mapping)
        df.drop(columns=['SEX'])
    return df
