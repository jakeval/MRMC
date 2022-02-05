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
NUM_TASKS = 12
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

    pois = np.random.choice(dataset[dataset.Y == -1].index, size=p['num_trials'])
    np.random.seed(p['seed'])

    d = dice_ml.Data(dataframe=dataset, continuous_features=preprocessor.continuous_features, outcome_name='Y')
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('tonumpy', ToNumpy()),
                          ('classifier', model)])
    backend = 'sklearn'
    m = dice_ml.Model(model=clf, backend=backend)
    dice = dice_ml.Dice(d, m, method="random")

    column_names_per_feature = [] # list of lists
    for feature in dataset.columns.difference(['Y']):
        column_names_per_feature.append(preprocessor.get_feature_names_out([feature]))

    experiment_immutable_feature_names = None
    if p['dataset'] == 'adult_income':
        experiment_immutable_feature_names = ['age', 'race', 'sex']
    elif p['dataset'] == 'german_credit':
        experiment_immutable_feature_names = ['age', 'sex']

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

    k_paths = p['k_paths']
    immutable_features = None
    feature_tolerances = None
    immutable_column_indices = None
    if p['immutable_features']:
        if p['dataset'] == 'adult_income':
            immutable_features = ['age', 'race', 'sex']
        if p['dataset'] == 'german_credit':
            immutable_features = ['age', 'sex']
        X = preprocessor.transform(dataset)
        immutable_columns = preprocessor.get_feature_names_out(immutable_features)
        immutable_column_indices = np.arange(X.columns.shape[0])[X.columns.isin(immutable_columns)]
        # feature_tolerances = {'age': 5}
    
    max_iterations = p['max_iterations']

    weight_function = lambda dir: scale_weight_function(dir, immutable_column_indices, p['weight_function_alpha'])
    perturb_dir = None
    if p['perturb_dir_random_scale'] is not None:
        perturb_dir = lambda dir: utils.random_perturb_dir(p['perturb_dir_random_scale'], dir, immutable_column_indices)

    num_trials = p['num_trials']
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
                              clf,
                              pois,
                              perturb_dir=perturb_dir,
                              immutable_features=immutable_features,
                              feature_tolerances=feature_tolerances)

    _, aggregated_stats = test.run_test()
    return aggregated_stats


def scale_weight_function(dir, immutable_column_indices, rescale_factor):
    new_dir = dir * rescale_factor
    new_dir[:,immutable_column_indices] = dir[:,immutable_column_indices]
    return new_dir


def get_params(num_trials, dataset_str):
    """Create a dataframe of input parameters.
    
    Each row of the dataframe contains a setting over parameters used by test_launcher to launch a test.
    Grid search is performed over these parameters, testing every combination of values for all the parameters.
    Simple parameters can be searched over naively. Parameters like early stopping are handled more carefully --
    if early stopping is disabled (early_stopping = False), there is no point in searching over early_stopping_cutoff.
    """

    # simple parameters have no conflicts
    params = {
        'num_trials': [num_trials],
        'max_iterations': [2],
        'weight_function_alpha': [0.7],
        'perturb_dir_random_scale': [None, 0.25, 0.5, 1, 2, 4],
        'k_paths': [4],
        'model': ['svc', 'random_forest'],
        'immutable_features': [True],
        'dataset': [dataset_str],
        'certainty_cutoff': [0.7]
    }
    
    return list(ParameterGrid(params))

def write_dataframe(params_df, results_dataframe_list, output_file):
    results_dataframe = pd.concat(results_dataframe_list, axis=0).reset_index()
    results_dataframe = results_dataframe
    final_df = pd.concat([params_df.reset_index(), results_dataframe], axis=1)
    final_df.to_pickle(output_file)
    if RUN_LOCALLY:
        outputs = [
            'model',
            'perturb_dir_random_scale',
            'certainty_cutoff',
            'Positive Probability (mean)',
            #'Model Certainty (mean)',
            'Final Point Distance (mean)',
            'Path Count (mean)',
            #'Point Sparsity (mean)',
        ]
        print(final_df[outputs])

def run_experiment():
    print("starting the script...")
    args = sys.argv
    dataset = None
    num_tests = int(args[2])
    dataset = args[1]
    print("dataset is ", dataset)
    output_file = os.path.join(OUTPUT_DIR, f'{dataset}.pkl')
    num_trials = 60

    models = {
        ('svc', 'german_credit'): model_utils.load_model('svc', 'german_credit'),
        ('svc', 'adult_income'): model_utils.load_model('svc', 'adult_income'),
        ('random_forest', 'german_credit'): model_utils.load_model('random_forest', 'german_credit'),
        ('random_forest', 'adult_income'): model_utils.load_model('random_forest', 'adult_income'),
    }

    german_data, _, german_preprocessor = da.load_german_credit_dataset()
    adult_data, _, adult_preprocessor = da.load_adult_income_dataset()

    all_params = get_params(num_trials, dataset)
    print(len(all_params))
    if num_tests == 0:
        num_tests = len(all_params)
    print(f"Run {num_tests} tests")
    params = all_params[:num_tests]
    for param_dict in params:
        new_params = {
            'seed': np.random.randint(999999),
            'dataset_payload': german_data if param_dict['dataset'] == 'german_credit' else adult_data,
            'preprocessor_payload': german_preprocessor if param_dict['dataset'] == 'german_credit' else adult_preprocessor,
            'model_payload': models[(param_dict['model'], param_dict['dataset'])]
        }
        param_dict.update(new_params)

    results = None
    with multiprocessing.Pool(NUM_TASKS) as p:
        results = p.map(test_launcher, params)

    param_df = pd.DataFrame(params).drop(['dataset_payload', 'preprocessor_payload', 'model_payload'], axis=1)

    write_dataframe(param_df, results, output_file)
    print("Finished experiment.")


if __name__ == '__main__':
    run_experiment()
