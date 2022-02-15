import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os

from experiments import stat_functions
from data import data_adapter as da
from models import model_utils

plt.style.use('style.mplstyle')


def load_data(directory):
    if not os.path.exists(f'{directory}/paths.csv'):
        df1 = pd.read_pickle(f'{directory}/adult_income.pkl')
        df2 = pd.read_pickle(f'{directory}/german_credit.pkl')
        pd.concat([df1, df2], ignore_index=True).to_csv(f'{directory}/paths.csv', index=False)
    df = pd.read_csv(f'{directory}/paths.csv')
    return df


def count_path_successes(paths, preprocessor, model, cutoff=0.7):
    successes = 0
    for path in paths:
        point = preprocessor.transform(path.iloc[[-1]])
        is_positive = model.predict_proba(point.to_numpy())[0,1] >= cutoff
        if is_positive:
            successes += 1
    return successes


def distance_comparison(baseline_paths, paths, preprocessor):
    avg_distance = 0
    for baseline_path, path in zip(baseline_paths, paths):
        point = preprocessor.transform(path.iloc[[-1]])
        baseline_point = preprocessor.transform(baseline_path.iloc[[-1]])
        diff = point.to_numpy() - baseline_point.to_numpy()
        avg_distance += np.linalg.norm(diff)
    return avg_distance / len(baseline_paths)


def get_stats_per_poi(df, param_columns, point_columns, stat_dict):
    """
    Returns a dataframe like:

    [param1, param2, ..., poi_index, poi_stat1, poi_stat2]
    """
    print("Get POI stats")
    result_df = pd.DataFrame(columns=(param_columns + list(stat_dict.keys()) + ['poi_index']))
    for params, param_df in df.groupby(param_columns):
        print(params)
        for poi, poi_df in param_df.groupby('poi_index'):
            paths = []
            for path_index, path_df in poi_df.groupby('path_index'):
                paths.append(path_df[point_columns])
            stat_df = pd.DataFrame(columns=(param_columns + ['poi_index']), data=([list(params) + [poi]]))
            for stat, calculate_stat in stat_dict.items():
                stat_df.loc[:,stat] = calculate_stat(paths)
            result_df = pd.concat([result_df, stat_df], ignore_index=True)
    return result_df


def get_comparison_path(comparison_df, param_value_tuples):
    mask = np.ones(comparison_df.shape[0]).astype(np.bool8)
    for param, value in param_value_tuples:
        mask = mask & (comparison_df[param] == value)
    return comparison_df.loc[mask,:].copy()


def get_comparison_stats(comparison_df, df, comparison_param, param_columns, point_columns, stat_dict):
    print("Get comparison stats")
    result_df = pd.DataFrame(columns=(param_columns + list(stat_dict.keys()) + ['poi_index']))
    for params, param_df in df.groupby(param_columns):
        print(params)
        for poi, poi_df in param_df.groupby('poi_index'):
            paths = []
            comparison_paths = []
            for path_index, path_df in poi_df.groupby('path_index'):
                paths.append(path_df[point_columns])
                shared_params = list(filter(lambda pair: pair[0] != comparison_param, zip(param_columns, params)))
                shared_params = shared_params + [('poi_index', poi), ('path_index', path_index)]
                comparison_paths.append(get_comparison_path(comparison_df, shared_params)[point_columns])
            stat_df = pd.DataFrame(columns=(param_columns + ['poi_index']), data=([list(params) + [poi]]))
            for stat, calculate_stat in stat_dict.items():
                stat_df.loc[:,stat] = calculate_stat(comparison_paths, paths)
            result_df = pd.concat([result_df, stat_df], ignore_index=True)
    return result_df


def aggregate_stats(stats_df, param_columns, stat_columns):
    result_df = pd.DataFrame(columns=(param_columns + stat_columns))
    std_dev_columns = list(map(lambda column: f"{column} (std)", stat_columns))
    for params, param_df in stats_df.groupby(param_columns):
        means, std_devs = param_df[stat_columns].mean(), param_df[stat_columns].std()
        stat_df = pd.DataFrame(columns=param_columns, data=[params])
        stat_df.loc[:,stat_columns] = means
        stat_df.loc[:,std_dev_columns] = std_devs
        result_df = pd.concat([result_df, stat_df], ignore_index=True)

    return result_df


def get_histogram(counts):
    plt.hist(counts)
    plt.show()


def get_all_poi_stats(df, model, preprocessor, param_columns, point_columns):
    column_names_per_feature = [] # list of lists
    for feature in point_columns:
        column_names_per_feature.append(preprocessor.get_feature_names_out([feature]))

    comparison_param = 'perturb_dir_random_scale'
    baseline_df = df[df[comparison_param] == 0]

    num_columns = df[point_columns].select_dtypes(include=np.number).columns.difference(['Y'])
    cat_columns = list(filter(lambda col: col not in num_columns, point_columns))

    stats = {
        'Iterations': stat_functions.check_path_count,
        'Path Length': lambda paths: stat_functions.check_path_length(preprocessor, paths),
        'Positive Probability': lambda paths: stat_functions.check_positive_probability(preprocessor, model, paths, 0.7),
        'Path Success Count': lambda paths: stat_functions.count_path_successes(paths, preprocessor, model),
        'Final Point Distance': lambda paths: stat_functions.check_final_point_distance(preprocessor, paths),
        'Sparsity': lambda paths: stat_functions.check_sparsity(paths, num_columns, cat_columns),
        'Diversity': lambda paths: stat_functions.check_diversity(preprocessor, paths),
    }

    comparison_stats = {
        'Comparison Distance': lambda df1, df2: stat_functions.distance_comparison(df1, df2, preprocessor),
        #'Comparison Similarity': lambda df1, df2: stat_functions.cosine_comparison(df1, df2, preprocessor)
    }


    poi_results = get_stats_per_poi(df,
                                  param_columns, point_columns,
                                  stats)

    comparison_results = get_comparison_stats(baseline_df,
                                            df,
                                            comparison_param,
                                            param_columns,
                                            point_columns,
                                            comparison_stats)
    poi_results[list(comparison_stats.keys())] = comparison_results[list(comparison_stats.keys())]
    return poi_results


def clean_paths(df):
    df.loc[df['perturb_dir_random_scale'].isnull(),'perturb_dir_random_scale'] = 0
    if 'kde_rtol' in df.columns:
        df = df.drop('kde_rtol', axis=1)
    return df


def write_poi_stats(directory):
    # Adult Income
    print("load adult")
    d, _, p = da.load_adult_income_dataset()
    df = clean_paths(pd.read_pickle(f'{directory}/adult_income.pkl'))
    rf_model = model_utils.load_model('random_forest', 'adult_income')
    lr_model = model_utils.load_model('svc', 'adult_income')
    misc_columns = ['poi_index', 'path_index']
    param_columns = list(df.columns.difference(d.columns).difference(misc_columns))
    point_columns = list(d.columns.difference(['Y']))
    print("get RF stats")
    adult_rf_results = get_all_poi_stats(df[df['model'] == 'random_forest'], rf_model, p, param_columns, point_columns)
    print(adult_rf_results)
    print("get LR stats")
    adult_lr_results = get_all_poi_stats(df[df['model'] == 'svc'], lr_model, p, param_columns, point_columns)
    print(adult_lr_results)

    # German Credit
    print("load german")
    d, _, p = da.load_german_credit_dataset()
    df = clean_paths(pd.read_pickle(f'{directory}/german_credit.pkl'))
    rf_model = model_utils.load_model('random_forest', 'german_credit')
    lr_model = model_utils.load_model('svc', 'german_credit')
    misc_columns = ['poi_index', 'path_index']
    param_columns = list(df.columns.difference(d.columns).difference(misc_columns))
    print("At start ", param_columns)
    point_columns = list(d.columns.difference(['Y']))
    print("get RF stats")
    german_rf_results = get_all_poi_stats(df[df['model'] == 'random_forest'], rf_model, p, param_columns, point_columns)
    print(german_rf_results)
    print("get LR stats")
    german_lr_results = get_all_poi_stats(df[df['model'] == 'svc'], lr_model, p, param_columns, point_columns)
    print(german_lr_results)

    #path_stats = pd.concat([adult_rf_results, adult_lr_results, german_rf_results, german_lr_results], ignore_index=True)
    #path_stats.to_csv(f'{directory}/poi_stats.csv', index=False)
    #print(path_stats)


def aggregate_stats(stats_df, param_columns, stat_columns):
    result_df = pd.DataFrame(columns=(param_columns + stat_columns))
    std_dev_columns = list(map(lambda column: f"{column} (std)", stat_columns))
    for params, param_df in stats_df.groupby(param_columns):
        print(params)
        means, std_devs = param_df[stat_columns].mean(), param_df[stat_columns].std()
        stat_df = pd.DataFrame(columns=param_columns, data=[params])
        stat_df.loc[:,stat_columns] = means
        stat_df.loc[:,std_dev_columns] = std_devs
        result_df = pd.concat([result_df, stat_df], ignore_index=True)

    return result_df


def write_aggregated_stats(directory):
    df = pd.read_csv(f'{directory}/poi_stats.csv')

    stat_columns = [
        'Iterations',
        'Path Length',
        'Positive Probability',
        'Path Success Count',
        'Final Point Distance',
        'Sparsity',
        'Diversity',
        'Comparison Distance'
    ]

    param_columns = list(df.columns.difference(stat_columns + ['poi_index']))

    results = aggregate_stats(df, param_columns, stat_columns)
    results.to_csv(f'{directory}/results.csv', index=False)
    print(results)


if __name__ == '__main__':
    write_poi_stats('../face_path_output')
    #write_aggregated_stats('../face_path_output')