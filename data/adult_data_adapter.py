import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import os


class AdultPreprocessor:
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
def load_data(data_dir='../data/adult'):
    train_filename = 'adult.data'
    test_filename = 'adult.test'
    if not check_for_reformatted_data(data_dir, train_filename, test_filename):
        train_filename, test_filename = reformat_data(data_dir, train_filename), reformat_data(data_dir, test_filename, is_test=True)
    else:
        train_filename, test_filename = get_reformatted_filename(train_filename), get_reformatted_filename(test_filename)
    
    data_df, test_df = load_and_process_data(data_dir, train_filename), load_and_process_data(data_dir, test_filename)
    
    category_features = ['workclass', 'occupation', 'race', 'sex', 'marital-status', 'education']
    continuous_features = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
    data_df = data_df.drop('education-num', axis=1)
    test_df = test_df.drop('education-num', axis=1)
    return data_df, test_df, AdultPreprocessor(category_features, continuous_features).fit(data_df)


def get_education_ordering(df):
    category_ordering = []
    for ed_num in np.sort(df['education-num'].unique()):
        ed_category = df[df['education-num'] == ed_num]['education'].unique()[0]
        category_ordering.append(ed_category)
    return category_ordering


# TODO: actually check for previously reformatted data
def check_for_reformatted_data(data_dir, train_filename, test_filename):
    train_reformatted = get_reformatted_filename(train_filename)
    return os.path.exists(os.path.join(data_dir, train_reformatted))


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
    new_df['Y'] = 1
    new_df.loc[new_df['income'] == '<=50K', 'Y'] = -1
    new_df = new_df.drop('income', axis=1)
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

    df = df.drop(columns=['fnlwgt', 'native-country', 'relationship'])
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
