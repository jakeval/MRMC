import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import os


class AdultPreprocessor:
    def __init__(self, categorical_features, ordinal_features, continuous_features):
        self.categorical_features = categorical_features
        self.ordinal_features = ordinal_features
        self.continuous_features = continuous_features

        self.sc_dict = None
        self.le_dict = None
        self.ohe_dict = None

    def fit(self, dataset):
        self.sc_dict = {}
        self.le_dict = {}
        self.ohe_dict = {}
        for feature in self.continuous_features:
            sc = StandardScaler()
            sc.fit(dataset[[feature]])
            self.sc_dict[feature] = sc
        for feature in self.ordinal_features:
            le = LabelEncoder()
            le.fit(dataset[feature])
            self.le_dict[feature] = le
        for feature in self.categorical_features:
            ohe = OneHotEncoder()
            ohe.fit(dataset[[feature]])
            self.ohe_dict[feature] = ohe
        return self

    def transform(self, dataset):
        df = dataset.copy()
        for feature in self.continuous_features:
            if feature in df.columns:
                df[feature] = self.sc_dict[feature].transform(df[[feature]])
        for feature in self.ordinal_features:
            if feature in df.columns:
                df[feature] = self.le_dict[feature].transform(df[feature])
        for feature in self.categorical_features:
            if feature in df.columns:
                ohe = self.ohe_dict[feature]
                feature_columns = ohe.get_feature_names_out([feature])
                df[feature_columns] = ohe.transform(df[[feature]]).toarray()
                df = df.drop(feature, axis=1)
        return df

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)

    # TODO: Shift the floor/ceil logic into MRM
    def inverse_transform(self, dataset):
        df = dataset.copy()
        for feature in self.continuous_features:
            if feature in df.columns:
                df[feature] = self.sc_dict[feature].inverse_transform(df[[feature]])
        for feature in self.ordinal_features:
            if feature in df.columns:
                df[feature] = df[feature].round().astype('int32')
                df[feature] = self.le_dict[feature].inverse_transform(df[feature])
        for feature in self.categorical_features:
            ohe = self.ohe_dict[feature]
            feature_columns = ohe.get_feature_names_out([feature])
            if df.columns.intersection(feature_columns).any():
                df[feature] = ohe.inverse_transform(df[feature_columns])
                df = df.drop(feature_columns, axis=1)
        return df


# TODO: support loading from online directly
def load_data(data_dir='../data/adult'):
    train_filename = 'adult.data'
    test_filename = 'adult.test'
    if not check_for_reformatted_data(data_dir, train_filename, test_filename):
        train_filename, test_filename = reformat_data(data_dir, train_filename), reformat_data(data_dir, test_filename, is_test=True)
    else:
        train_filename, test_filename = get_reformatted_filename(train_filename), get_reformatted_filename(test_filename)
    
    data_df, test_df = load_and_process_data(data_dir, train_filename), load_and_process_data(data_dir, test_filename)
    
    category_features = ['workclass', 'occupation', 'race']
    ordinal_features = ['sex', 'marital-status', 'education']
    continuous_features = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
    return data_df, test_df, AdultPreprocessor(category_features, ordinal_features, continuous_features).fit(data_df)


# TODO: actually check for previously reformatted data
def check_for_reformatted_data(data_dir, train_filename, test_filename):
    False


def get_reformatted_filename(filename):
    return f"{filename}.reformatted"


def reformat_data(data_dir, filename, is_test=False):
    reader = None
    new_filename = get_reformatted_filename(filename)
    column_names = "age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income".split(',')
    with open(os.path.join(data_dir, filename), newline='') as file:
        reader = csv.reader(file, delimiter=',')
        with open(os.path.join(data_dir, new_filename), 'w', newline='') as newfile:
            writer = csv.writer(newfile, delimiter=',')
            writer.writerow(column_names)
            for i, row in enumerate(reader):
                if is_test and i == 0:
                    continue
                stripped_row = list(map(lambda s: s.strip(), row))
                if is_test:
                    stripped_row = list(map(lambda s: s.rstrip('.'), stripped_row))
                writer.writerow(stripped_row)
    return new_filename


def convert_label(dataframe):
    new_df = dataframe.copy()
    new_df['income'] = dataframe['income'].mask(dataframe['income'] == '>50K', 1)
    new_df['income'] = new_df['income'].mask(dataframe['income'] == '<=50K', -1)
    new_df = new_df.rename(columns = {'income': 'Y'})
    return new_df


def recategorize_feature(column, inverse_category_dict):
    new_column = column.copy()
    for key, val_list in inverse_category_dict.items():
        for val in val_list:
            new_column = np.where(new_column == val, key, new_column)
    return new_column


def load_and_process_data(data_dir, filename):
    df = None
    with open(os.path.join(data_dir, filename)) as f:
        df = pd.read_csv(f)

    df = df.drop(columns=['education-num', 'fnlwgt', 'native-country', 'relationship'])
    df = df.drop_duplicates()

    for c in df.columns:
        df = df.drop(index=df[df[c] == '?'].index)

    df = convert_label(df)
    new_categories_dict = {
        'Single': ['Never-married', 'Divorced', 'Separated', 'Widowed'],
        'Married': ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse']
    }
    df['marital-status'] = recategorize_feature(df['marital-status'], new_categories_dict)
    return df
