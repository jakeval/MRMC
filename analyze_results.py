import numpy as np
import pandas as pd
import os
import sys

import multiprocessing

from experiments import stat_functions
from data import data_adapter as da
from models import model_utils

sys.path.append(os.getcwd())

NUM_TASKS = 48

RUN_LOCALLY = False
BASE_DIR = '/mnt/nfs/home/jasonvallada'
if RUN_LOCALLY:
    BASE_DIR = '..'
    NUM_TASKS = 1


def write_dataframe(results, directory, dataset):
    result_df = pd.concat(results, ignore_index=True)
    result_df.to_csv(f'{directory}/{dataset}_stats.csv', index=False)
    if RUN_LOCALLY:
        print(result_df[['k_dirs', 'poi_index']])


def run_analysis():
    print("starting the script...")
    args = sys.argv
    dataset = args[1]
    method = args[2]
    directory = f'{BASE_DIR}/{method}_path_output'
    print(dataset, method)

    params = get_params(directory, dataset)
    results = None
    with multiprocessing.Pool(NUM_TASKS) as p:
        results = p.map(collect_stats, params)
    write_dataframe(results, directory, dataset)


def get_params(directory, dataset):
    df = pd.read_pickle(f'{directory}/{dataset}.pkl')
    d, p = None, None
    if dataset == 'adult_income':
        d, _, p = da.load_adult_income_dataset()
    elif dataset == 'german_credit':
        d, _, p = da.load_german_credit_dataset()

    misc_columns = ['path_index', 'path_order', 'trial_key']
    point_columns = list(d.columns.difference(['Y']))
    param_columns = list(df.columns.difference(d.columns).difference(misc_columns))

    num_columns = list(df[point_columns].select_dtypes(include=np.number).columns)
    cat_columns = list(filter(lambda col: col not in num_columns, point_columns))

    shared_params = {
        'preprocessor_payload': p,
        'num_columns': num_columns,
        'cat_columns': cat_columns,
        'point_columns': point_columns,
        'param_columns': param_columns,
    }

    param_dicts = []
    for trial_key in df['trial_key'].unique():
        param_dict = {}
        param_dict.update(shared_params)

        paths_df = df[df['trial_key'] == trial_key]
        param_dict['paths_payload'] = paths_df

        model = paths_df['model'].unique()[0]
        param_dict['model_payload'] = model_utils.load_model(model, dataset)
        param_dicts.append(param_dict)

    return param_dicts


def collect_stats(p):
    preprocessor = p['preprocessor_payload']
    model = p['model_payload']
    df = p['paths_payload']

    num_columns = p['num_columns']
    cat_columns = p['cat_columns']
    point_columns = p['point_columns']
    param_columns = p['param_columns']

    # poi_index, path_index, path_order, trial_key

    stat_dict = {
        'Iterations': stat_functions.check_path_count,
        'Path Length': lambda paths: stat_functions.check_path_length(preprocessor, paths),
        'Positive Probability': lambda paths: stat_functions.check_positive_probability(preprocessor, model, paths, 0.7),
        'Path Success Count': lambda paths: stat_functions.count_path_successes(paths, preprocessor, model),
        'Final Point Distance': lambda paths: stat_functions.check_final_point_distance(preprocessor, paths),
        'Sparsity': lambda paths: stat_functions.check_sparsity(paths, num_columns, cat_columns),
        'Diversity': lambda paths: stat_functions.check_diversity(preprocessor, paths),
    }

    comparison_stat_dict = {
        #'Comparison Distance': lambda df1, df2: stat_functions.distance_comparison(df1, df2, preprocessor),
        #'Comparison Similarity': lambda df1, df2: stat_functions.cosine_comparison(df1, df2, preprocessor)
    }

    stat_columns = list(stat_dict.keys()) + list(comparison_stat_dict.keys())

    paths = []

    for path_idx in df['path_index'].unique():
        path_df = df[df['path_index'] == path_idx].sort_values('path_order')
        path_points = path_df[point_columns]
        paths.append(path_points)

    stat_df = pd.DataFrame(columns=stat_columns, data=np.zeros((1, len(stat_columns))))
    for stat, calculate_stat in stat_dict.items():
        stat_df.loc[:,stat] = calculate_stat(paths) # returns a single stat

    stat_df[param_columns] = df.loc[df.index[0],param_columns]

    return stat_df


def aggregate_stats(stats_df, param_columns, stat_columns):
    result_df = pd.DataFrame(columns=(param_columns + stat_columns))
    std_dev_columns = list(map(lambda column: f"{column} (std)", stat_columns))
    for params, param_df in stats_df.groupby(param_columns):
        #print(params)
        #print(param_df)
        means, std_devs = param_df[stat_columns].mean(), param_df[stat_columns].std()
        stat_df = pd.DataFrame(columns=param_columns, data=[params])
        stat_df[stat_columns] = means
        stat_df[std_dev_columns] = std_devs
        result_df = pd.concat([result_df, stat_df], ignore_index=True)

    return result_df


def write_aggregated_stats(directory):
    df_adult = pd.read_csv(f'{directory}/adult_income_stats.csv').drop('Unnamed: 0', axis=1)
    df_german = pd.read_csv(f'{directory}/german_credit_stats.csv').drop('Unnamed: 0', axis=1)
    if 'Unnamed: 0' in df_adult.columns:
        df_adult = df_adult.drop('Unnamed: 0', axis=1)
    if 'Unnamed: 0' in df_german.columns:
        df_german = df_german.drop('Unnamed: 0', axis=1)

    df = pd.concat([df_adult, df_german], ignore_index=True)

    stat_columns = [
        'Iterations',
        'Path Length',
        'Positive Probability',
        'Path Success Count',
        'Final Point Distance',
        'Sparsity',
        'Diversity',
        #'Comparison Distance'
    ]

    param_columns = list(df.columns.difference(stat_columns + ['poi_index']))

    results = aggregate_stats(df, param_columns, stat_columns)
    results.to_csv(f'{directory}/results.csv', index=False)
    print(results)


if __name__ == '__main__':
    #write_aggregated_stats('../dice_path_output')
    run_analysis()
