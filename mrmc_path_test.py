import numpy as np
from data import data_adapter as da
from models import random_forest
from experiments.test_mrmc import MrmcTestRunner
from core.mrmc import MRM, MRMCIterator, MRMIterator
from experiments import path_stats, point_stats
from core import utils

import multiprocessing

import pandas as pd
from sklearn.model_selection import ParameterGrid
import itertools
import sys
from models import model_utils
import os

sys.path.append(os.getcwd())
np.random.seed(885577)

NUM_TASKS = 48

RUN_LOCALLY = False
OUTPUT_DIR = '/mnt/nfs/home/jasonvallada/mrmc_path_output'
if RUN_LOCALLY:
    OUTPUT_DIR = '../mrmc_path_output'
    NUM_TASKS = 1

def test_launcher(p):
    model = p['model_payload']
    preprocessor = p['preprocessor_payload']
    dataset = p['dataset_payload']

    np.random.seed(p['poi_seed'])
    poi_indices = np.random.choice(dataset[dataset.Y == -1].index, size=p['num_trials'])
    pois = dataset.loc[poi_indices].drop('Y', axis=1)
    np.random.seed(p['seed'])

    X = np.array(preprocessor.transform(dataset.drop('Y', axis=1)))
    model_scores = model.predict_proba(X)
    dataset = da.filter_from_model(dataset, model_scores)

    num_trials = p['num_trials']
    k_dirs = p['k_dirs']
    max_iterations = p['max_iterations']
    experiment_immutable_feature_names = p['experiment_immutable_features']
    validate = p['validate']
    early_stopping = None

    immutable_column_names = None
    immutable_features = None
    feature_tolerances = None
    if p['immutable_features']:
        immutable_features = experiment_immutable_feature_names
        immutable_column_names = preprocessor.get_feature_names_out(immutable_features)
        feature_tolerances = {
            'age': 5
        }

    perturb_dir = None
    if p['perturb_dir_random_scale'] is not None:
        perturb_dir = lambda dir: utils.random_perturb_dir(p['perturb_dir_random_scale'], dir)
    if p['sparsity']:
        if perturb_dir is None:
            perturb_dir = lambda dir: utils.priority_dir(dir, k=5)
        else:
            original_perturbation = perturb_dir
            perturb_dir = lambda dir: utils.priority_dir(original_perturbation(dir), k=5)

    if p['early_stopping']:
        early_stopping = lambda point: utils.model_early_stopping(model, point, cutoff=p['early_stopping_cutoff'])
    weight_function = None
    if p['weight_function'] == 'centroid':
        weight_function = lambda dir, poi, X: utils.centroid_normalization(dir, poi, X, alpha=p['weight_centroid_alpha'])
    alpha_function = None
    if p['alpha_function'] == 'volcano':
        alpha_function = lambda dist: utils.volcano_alpha(dist, cutoff=p['alpha_volcano_cutoff'], degree=p['alpha_volcano_degree'])
    elif p['alpha_function'] == 'normal':
        alpha_function = lambda dist: utils.normal_alpha(dist, width=p['alpha_normal_width'])

    mrm = MRM(alpha=alpha_function, weight_function=weight_function, perturb_dir=perturb_dir, immutable_column_names=immutable_column_names)
    mrmc = MRMCIterator(k_dirs, mrm, preprocessor, max_iterations, early_stopping=early_stopping, validate=validate)

    column_names_per_feature = [] # list of lists
    for feature in dataset.columns.difference(['Y']):
        column_names_per_feature.append(preprocessor.get_feature_names_out([feature]))

    path_statistics = {
        'Path Invalidity': lambda paths: path_stats.check_validity_distance(preprocessor, paths),
        'Path Count': path_stats.check_path_count,
        'Path Length': path_stats.check_path_length,
        'Path Immutable Violations': lambda paths: path_stats.check_immutability(preprocessor, experiment_immutable_feature_names, paths),
        'Average Path Sparsity': lambda paths: path_stats.check_sparsity(preprocessor, paths),
        'Path Invalidity': lambda paths: path_stats.check_validity(preprocessor, column_names_per_feature, paths),
        'Diversity': path_stats.check_diversity
    }
    point_statistics = {
        'Positive Probability': lambda poi, paths: point_stats.check_positive_probability(model, poi, paths, p['early_stopping_cutoff']),
        'Model Certainty': lambda poi, paths: point_stats.check_model_certainty(model, poi, paths),
        'Point Invalidity': lambda poi, points: point_stats.check_validity(preprocessor, column_names_per_feature, poi, points),
        'Final Point Distance': point_stats.check_final_point_distance,
        'Point Immutable Violations': lambda poi, points: point_stats.check_immutability(preprocessor, experiment_immutable_feature_names, poi, points),
        'Point Sparsity': lambda poi, points: point_stats.check_sparsity(preprocessor, poi, points),
    }
    cluster_statistics = {
        'Cluster Size': path_stats.check_cluster_size,
    }

    test = MrmcTestRunner(num_trials, dataset, preprocessor, mrmc, path_statistics,
                        point_statistics, cluster_statistics, pois, immutable_features=immutable_features,
                        immutable_strict=False, feature_tolerances=feature_tolerances)
    stats, aggregated_stats = test.run_test()
    return aggregated_stats


def get_params(num_trials, dataset_str):
    """Create a dataframe of input parameters.
    
    Each row of the dataframe contains a setting over parameters used by test_launcher to launch a test.
    Grid search is performed over these parameters, testing every combination of values for all the parameters.
    Simple parameters can be searched over naively. Parameters like early stopping are handled more carefully --
    if early stopping is disabled (early_stopping = False), there is no point in searching over early_stopping_cutoff.
    """

    # simple parameters have no conflicts
    simple_params = {
        'num_trials': [num_trials],
        'k_dirs': [4],
        'max_iterations': [15],
        'validate': [True],
        'sparsity': [True],
        'model': ['random_forest', 'svc'],
        'perturb_dir_random_scale': [None, 0.25, 0.5, 1, 2, 4]
    }

    dataset = []
    if dataset_str == 'adult_income':
        dataset.append(
            {
                'dataset': ['adult_income'],
                'experiment_immutable_features': [['age', 'sex', 'race']],
                'immutable_features': [True],
            })
    else:
        dataset.append(
            {
                'dataset': ['german_credit'],
                'experiment_immutable_features': [['age', 'sex']],
                'immutable_features': [True],
            })

    early_stopping = [
        {
            'early_stopping': [True],
            'early_stopping_cutoff': [0.7],
        }
    ]

    weight_function = [
        {
            'weight_function': ['centroid'],
            'weight_centroid_alpha': [0.7]
        },
    ]

    alpha_function = [
        {
            'alpha_function': ['volcano'],
            'alpha_volcano_cutoff': [0.2],
            'alpha_volcano_degree': [2, 4, 8]
        },
        {
            'alpha_function': ['normal'],
            'alpha_normal_width': [0.5, 2]
        }
    ]

    # the simple and constrained parameters are combined into a list of dictionaries which respect the parameter constraints
    constrained_params = [dataset, early_stopping, weight_function, alpha_function]
    params = []
    for dict_tuple in itertools.product(*constrained_params):
        d = {}
        for dict in dict_tuple:
            d.update(dict)
        d.update(simple_params)
        params.append(d)
    
    return list(ParameterGrid(params))


def write_dataframe(params_df, results_dataframe_list, output_file):
    results_dataframe = pd.concat(results_dataframe_list, axis=0).reset_index()
    results_dataframe = results_dataframe
    final_df = pd.concat([params_df, results_dataframe], axis=1)
    final_df.to_pickle(output_file)


def run_experiment():
    print("starting the script...")
    args = sys.argv
    dataset = args[1]
    num_tests = int(args[2])
    output_file = os.path.join(OUTPUT_DIR, f'{dataset}.pkl')
    print("dataset is ", dataset)

    num_trials = 30

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

    poi_seed = 148294

    all_params = get_params(num_trials, dataset)
    print(len(all_params))
    if num_tests == 0:
        num_tests = len(all_params)
    print(f"Run {num_tests} tests")

    params = all_params[:num_tests]
    for param_dict in params:
        new_params = {
            'poi_seed': poi_seed,
            'seed': np.random.randint(999999),
            'dataset_payload': dataset_payload,
            'preprocessor_payload': preprocessor_payload,
            'model_payload': models[(param_dict['model'], param_dict['dataset'])],
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
