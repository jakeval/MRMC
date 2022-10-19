import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import os


# @TODO(jakeval): Dataset loading is refactored as-needed. Either remove this or
#                 refactor it once it is needed.
class GermanPreprocessor:
    def __init__(self, categorical_features, continuous_features):
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features

        self.sc_dict = None
        self.ohe_dict = None

    def fit(self, dataset):
        self.sc_dict = {}
        self.ohe_dict = {}
        for feature in self.continuous_features:
            sc = StandardScaler()
            sc.fit(dataset[[feature]])
            self.sc_dict[feature] = sc
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
        for feature in self.categorical_features:
            ohe = self.ohe_dict[feature]
            feature_columns = ohe.get_feature_names_out([feature])
            if df.columns.intersection(feature_columns).any():
                df[feature] = ohe.inverse_transform(df[feature_columns])
                df = df.drop(feature_columns, axis=1)
        return df

    def get_feature_names_out(self, features):
        features_out = []
        for feature in features:
            if feature in self.categorical_features:
                features_out += list(self.ohe_dict[feature].get_feature_names_out([feature]))
            else:
                features_out.append(feature)
        return features_out


# TODO: support loading from online directly
def load_data(data_dir='../data/german'):
    data_filename = 'german.data'
    train_filename, test_filename = get_filenames()
    if not reformatted_files_exist(os.path.join(data_dir, train_filename)):
        new_filename = reformat_data(data_dir, data_filename)
        train_filename, test_filename = write_train_test(data_dir, new_filename)
    data_df, test_df = load_and_process_data(data_dir, train_filename), load_and_process_data(data_dir, test_filename)
    
    continuous_features = ['duration', 'credit-amount', 'installment-rate', 'residence-duration', 'age', 'num-credits', 'num-liable']
    category_features = data_df.columns.difference(continuous_features).difference(['Y'])
    return data_df, test_df, GermanPreprocessor(category_features, continuous_features).fit(data_df)


def reformatted_files_exist(train_filename):
    return os.path.exists(train_filename)


def get_filenames():
    return "german.train", "german.test"


def write_train_test(data_dir, file):
    df = pd.read_csv(os.path.join(data_dir, file))
    train_df = df.sample(frac=0.8)
    test_df = df[~df.index.isin(train_df.index)]
    train_filename, test_filename = f'german.train', f'german.test'
    train_df.to_csv(os.path.join(data_dir, train_filename), index=False)
    test_df.to_csv(os.path.join(data_dir, test_filename), index=False)
    return train_filename, test_filename


def get_education_ordering(df):
    category_ordering = []
    for ed_num in np.sort(df['education-num'].unique()):
        ed_category = df[df['education-num'] == ed_num]['education'].unique()[0]
        category_ordering.append(ed_category)
    return category_ordering


# TODO: actually check for previously reformatted data
def check_for_reformatted_data(data_dir, train_filename, test_filename):
    False


def get_reformatted_filename(filename):
    return f"{filename}.reformatted"


def reformat_data(data_dir, filename):
    reader = None
    new_filename = get_reformatted_filename(filename)
    column_names = "checking-status,duration,credit-history,loan-purpose,credit-amount,savings-account,employment-duration,installment-rate,personal-status,guarantors,residence-duration,property,age,installment-plan,housing,num-credits,job,num-liable,telephone,foreign-worker,Y".split(',')
    with open(os.path.join(data_dir, filename), newline='') as file:
        reader = csv.reader(file, delimiter=' ')
        with open(os.path.join(data_dir, new_filename), 'w', newline='') as newfile:
            writer = csv.writer(newfile, delimiter=',')
            writer.writerow(column_names)
            for i, row in enumerate(reader):
                stripped_row = list(map(lambda s: s.strip(), row))
                writer.writerow(stripped_row)
    return new_filename


def convert_label(dataframe):
    new_df = dataframe.copy()
    new_df['Y'] = dataframe['Y'].mask(dataframe['Y'] == 2, -1)
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

    df['sex'] = 'M'
    df.loc[(df['personal-status'] == 'A92') | (df['personal-status'] == 'A95'), 'sex'] = 'F'
    df['relationship'] = 'Not Single'
    df.loc[(df['personal-status'] == 'A93') | (df['personal-status'] == 'A95'), 'relationship'] = 'Single'
    df = df.drop('personal-status', axis=1)

    df = convert_label(df)
    return df
