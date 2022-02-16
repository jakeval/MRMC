import numpy as np
from data import data_adapter as da
from experiments.test_dice_path import DicePathTestRunner
from experiments import point_stats, path_stats
from models import model_utils
from core import utils
import dice_ml
from sklearn.pipeline import Pipeline

import multiprocessing
import pandas as pd
from sklearn.model_selection import ParameterGrid
import sys
import os

np.random.seed(88557)

RUN_LOCALLY = False
NUM_TASKS = 48
OUTPUT_DIR = '/mnt/nfs/home/jasonvallada/dice_path_output'
if RUN_LOCALLY:
    OUTPUT_DIR = '../dice_path_output'
    NUM_TASKS = 1

class ToNumpy:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def transform(self, X):
        return X.to_numpy()

def test_launcher(p):
    if RUN_LOCALLY:
        print("Begin Test")
    model = p['model_payload']
    preprocessor = p['preprocessor_payload']
    dataset = p['dataset_payload']

    np.random.seed(p['seed'])
    poi_index = p['poi_index']
    poi = dataset.loc[[poi_index]].drop('Y', axis=1)

    d = dice_ml.Data(dataframe=dataset, continuous_features=preprocessor.continuous_features, outcome_name='Y')
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('tonumpy', ToNumpy()),
                          ('classifier', model)])
    backend = 'sklearn'
    m = dice_ml.Model(model=clf, backend=backend)
    dice = dice_ml.Dice(d, m, method="random")

    get_positive_probability = lambda p: model.predict_proba(p.to_numpy())[0,1]

    column_names_per_feature = [] # list of lists
    for feature in dataset.columns.difference(['Y']):
        column_names_per_feature.append(preprocessor.get_feature_names_out([feature]))

    experiment_immutable_feature_names = None
    if p['dataset'] == 'adult_income':
        experiment_immutable_feature_names = ['age', 'race', 'sex']
    elif p['dataset'] == 'german_credit':
        experiment_immutable_feature_names = ['age', 'sex']

    k_paths = p['k_paths']
    immutable_features = None
    feature_tolerances = None
    immutable_column_indices = None
    if p['immutable_features']:
        if p['dataset'] == 'adult_income':
            immutable_features = ['age', 'race', 'sex']
        if p['dataset'] == 'german_credit':
            immutable_features = ['age', 'sex']
        X = preprocessor.transform(dataset).drop('Y', axis=1)
        immutable_columns = preprocessor.get_feature_names_out(immutable_features)
        immutable_column_indices = np.arange(X.columns.shape[0])[X.columns.isin(immutable_columns)]
        # feature_tolerances = {'age': 5}

    path_statistics = {
        'Path Count': path_stats.check_path_count,
        'Path Length': path_stats.check_path_length,
        'Path Immutable Violations': lambda paths: path_stats.check_immutability(preprocessor, experiment_immutable_feature_names, paths),
        'Average Path Sparsity': lambda paths: path_stats.check_sparsity(preprocessor, paths),
        'Path Invalidity': lambda paths: path_stats.check_validity(preprocessor, column_names_per_feature, paths),
    }
    point_statistics = {
        'Positive Probability': lambda poi, paths: point_stats.check_positive_probability(model, poi, paths, p['certainty_cutoff']),
        'Point Invalidity': lambda poi, points: point_stats.check_validity(preprocessor, column_names_per_feature, poi, points),
        'Final Point Distance': point_stats.check_final_point_distance,
        'Point Immutable Violations': lambda poi, points: point_stats.check_immutability(preprocessor, experiment_immutable_feature_names, poi, points),
        'Point Sparsity': lambda poi, points: point_stats.check_sparsity(preprocessor, poi, points),
        'Diversity': point_stats.check_diversity,
        'Model Certainty': lambda poi, cf_points: point_stats.check_model_certainty(model, poi, cf_points)
    }
    
    max_iterations = p['max_iterations']

    weight_function = lambda dir: utils.scale_weight_function(dir, p['weight_function_alpha'])

    perturb_dir = None
    if p['perturb_dir_random_scale'] is not None:
        num_features = dataset.select_dtypes(include=np.number).columns.difference(['Y'])
        cat_features = dataset.columns.difference(num_features).difference(['Y'])
        perturb_dir = lambda point, dir: utils.random_perturb_dir(
            preprocessor,
            p['perturb_dir_random_scale'],
            p['perturb_dir_random_scale'],
            point,
            dir,
            num_features,
            cat_features,
            immutable_features)

    num_trials = 1
    certainty_cutoff = p['certainty_cutoff']
    test = DicePathTestRunner(num_trials,
                              dataset,
                              preprocessor,
                              dice,
                              point_statistics,
                              path_statistics,
                              k_paths,
                              certainty_cutoff,
                              max_iterations,
                              weight_function,
                              get_positive_probability,
                              None,
                              perturb_dir=perturb_dir,
                              immutable_features=immutable_features,
                              feature_tolerances=feature_tolerances)

    stats, paths, poi = test.run_trial(poi)
    paths_indexed = []
    for path in paths:
        path = path.copy()
        path['path_order'] = np.arange(path.shape[0])
        paths_indexed.append(path)
    return paths_indexed


def get_params(dataset_str, dataset_poi_indices, seed):
    """Create a dataframe of input parameters.
    
    Each row of the dataframe contains a setting over parameters used by test_launcher to launch a test.
    Grid search is performed over these parameters, testing every combination of values for all the parameters.
    Simple parameters can be searched over naively. Parameters like early stopping are handled more carefully --
    if early stopping is disabled (early_stopping = False), there is no point in searching over early_stopping_cutoff.
    """

    # simple parameters have no conflicts
    params = {
        'seed': [seed],
        'num_trials': [1],
        'max_iterations': [15],
        'weight_function_alpha': [0.3, 0.7],
        'perturb_dir_random_scale': [None, 0.25, 0.5, 0.75, 1],
        'k_paths': [4],
        'model': ['svc', 'random_forest'],
        'immutable_features': [True],
        'dataset': [dataset_str],
        'certainty_cutoff': [0.7],
        'poi_index': dataset_poi_indices,
    }
    
    return list(ParameterGrid(params))

def clean_dict(d):
    d2 = {}
    for key, val in d.items():
        if key not in ['dataset_payload', 'preprocessor_payload', 'model_payload']:
            d2[key] = val
    return d2

def write_dataframe(columns, preprocessor, param_dicts, results_list, output_file):
    results_dataframes = []
    trial_key = 0
    for param_dict, paths in zip(param_dicts, results_list):
        paths_df = pd.DataFrame(columns=columns)
        for i, path in enumerate(paths):
            path_df = preprocessor.inverse_transform(path)
            path_df['path_index'] = i
            paths_df = pd.concat([paths_df, path_df], ignore_index=True)
        paths_df['trial_key'] = trial_key
        trial_key += 1
        param_dict = clean_dict(param_dict)
        keys = list(param_dict.keys())
        values = list(param_dict.values())
        paths_df.loc[:,keys] = values
        results_dataframes.append(paths_df)
    final_df = pd.concat(results_dataframes, ignore_index=True)
    final_df.to_pickle(output_file)


def run_experiment():
    print("starting the script...")
    args = sys.argv
    dataset = None
    num_tests = int(args[2])
    dataset = args[1]
    print("dataset is ", dataset)
    output_file = os.path.join(OUTPUT_DIR, f'{dataset}.pkl')

    num_trials = 100
    if RUN_LOCALLY:
        num_trials = 2

    models = {
        ('svc', 'german_credit'): model_utils.load_model('svc', 'german_credit'),
        ('svc', 'adult_income'): model_utils.load_model('svc', 'adult_income'),
        ('random_forest', 'german_credit'): model_utils.load_model('random_forest', 'german_credit'),
        ('random_forest', 'adult_income'): model_utils.load_model('random_forest', 'adult_income'),
    }

    dataset_payload, preprocessor_payload = None, None
    if dataset == 'adult_income':
        dataset_payload, _, preprocessor_payload = da.load_adult_income_dataset()
    elif dataset == 'german_credit':
        dataset_payload, _, preprocessor_payload = da.load_german_credit_dataset()
    else:
        print("No dataset recognized")
        return

    poi_indices = np.random.choice(dataset_payload[dataset_payload.Y == -1].index, size=num_trials)
    seed = 885577

    all_params = get_params(dataset, poi_indices, seed)
    print(len(all_params))
    if num_tests == 0:
        num_tests = len(all_params)
    print(f"Run {num_tests} tests")

    params = all_params[:num_tests]
    for param_dict in params:
        new_params = {
            'dataset_payload': dataset_payload,
            'preprocessor_payload': preprocessor_payload,
            'model_payload': models[(param_dict['model'], param_dict['dataset'])]
        }
        param_dict.update(new_params)

    results = None
    with multiprocessing.Pool(NUM_TASKS) as p:
        results = p.map(test_launcher, params)
    columns = dataset_payload.columns.difference(['Y'])
    write_dataframe(columns, preprocessor_payload, params, results, output_file)
    print("Finished experiment.")


if __name__ == '__main__':
    run_experiment()
