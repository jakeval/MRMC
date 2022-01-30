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
np.random.seed(88557)

NUM_TASKS = 8

RUN_LOCALLY = False
SCRATCH_DIR = '/mnt/nfs/scratch1/jasonvallada'
OUTPUT_DIR = '/mnt/nfs/home/jasonvallada/corrections_output'
LOG_DIR = '/mnt/nfs/home/jasonvallada/MRMC/logs'
if RUN_LOCALLY:
    SCRATCH_DIR = '.'
    OUTPUT_DIR = '.'
    LOG_DIR = '.'


def test_launcher(p):
    if RUN_LOCALLY:
        print("Begin Test")
        print(p)
    np.random.seed(p['seed'])
    model = p['model_payload']
    preprocessor = p['preprocessor_payload']
    dataset = p['dataset_payload']

    X = np.array(preprocessor.transform(dataset.drop('Y', axis=1)))
    model_scores = model.predict_proba(X)
    dataset = da.filter_from_model(dataset, model_scores)

    num_trials = p['num_trials']
    k_dirs = p['k_dirs']
    max_iterations = p['max_iterations']
    experiment_immutable_feature_names = p['experiment_immutable_features']
    immutable_column_names = None
    immutable_features = None
    feature_tolerances = None
    immutable_strict = False
    if p['immutable_features'] is not None:
        immutable_column_names = preprocessor.get_feature_names_out(p['immutable_features'])
        immutable_features = p['immutable_features']
        feature_tolerances = {
            'age': 5
        }
        immutable_strict = p['immutable_strict']
    validate = p['validate']
    early_stopping = None

    perturb_dir = None
    if p['perturb_dir'] == 'random':
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
    }
    point_statistics = {
        'Positive Probability': lambda poi, paths: point_stats.check_positive_probability(model, poi, paths, p['early_stopping_cutoff']),
        'Point Invalidity': lambda poi, points: point_stats.check_validity(preprocessor, column_names_per_feature, poi, points),
        'Final Point Distance': point_stats.check_final_point_distance,
        'Point Immutable Violations': lambda poi, points: point_stats.check_immutability(preprocessor, experiment_immutable_feature_names, poi, points),
        'Point Sparsity': lambda poi, points: point_stats.check_sparsity(preprocessor, poi, points),
        'Diversity': point_stats.check_diversity
    }
    cluster_statistics = {
        'Cluster Size': path_stats.check_cluster_size,
    }

    test = MrmcTestRunner(num_trials, dataset, preprocessor, mrmc, path_statistics,
                        point_statistics, cluster_statistics, immutable_features=immutable_features, immutable_strict=immutable_strict, feature_tolerances=feature_tolerances)
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
        'validate': [True, False],
        'early_stopping': [True],
        'sparsity': [True, False],
        'model': ['svc', 'random_forest'],
        'early_stopping_cutoff': [0.6, 0.7],
        'weight_function': ['centroid'],
        'weight_centroid_alpha': [0.7],
        'immutable_strict': [False],
    }

    dataset = None
    if dataset_str == 'adult_income':
        dataset = [
            {
            'dataset': ['adult_income'],
            'experiment_immutable_features': [['age', 'sex', 'race']],
            'immutable_features': [['age', 'sex', 'race']]
            }
        ]
    elif dataset_str == 'german_credit':
        dataset = [
            {
            'dataset': ['german_credit'],
            'experiment_immutable_features': [['age', 'sex']],
            'immutable_features': [['age', 'sex']]
            }
        ]

    alpha_function = [
        {
            'alpha_function': ['volcano'],
            'alpha_volcano_cutoff': [0.2],
            'alpha_volcano_degree': [2, 4],
        },
        {
            'alpha_function': ['normal'],
            'alpha_normal_width': [0.5],
        },
    ]

    perturb_dir = [
        {
            'perturb_dir': ['random'],
            'perturb_dir_random_scale': [0.25, 0.5, 1, 2, 4],
        },
        {
            'perturb_dir': [None]
        }
    ]

    # the simple and constrained parameters are combined into a list of dictionaries which respect the parameter constraints
    constrained_params = [dataset, alpha_function, perturb_dir]
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
    final_df = pd.concat([params_df.reset_index(), results_dataframe], axis=1)
    final_df.to_pickle(output_file)


def run_experiment():
    print("starting the script...")
    args = sys.argv
    dataset = None
    num_tests = int(args[2])
    dataset = args[1]
    print("dataset is ", dataset)
    output_file = os.path.join(OUTPUT_DIR, f'{dataset}.pkl')
    num_trials = 30

    models = {
        ('svc', 'german_credit'): model_utils.load_model('svc', 'german_credit'),
        ('svc', 'adult_income'): model_utils.load_model('svc', 'adult_income'),
        ('random_forest', 'german_credit'): model_utils.load_model('random_forest', 'german_credit'),
        ('random_forest', 'adult_income'): model_utils.load_model('random_forest', 'adult_income'),
    }

    german_data, _, german_preprocessor = da.load_german_credit_dataset()
    adult_data, _, adult_preprocessor = da.load_adult_income_dataset()

    preprocessors = {
        'german_credit': german_preprocessor,
        'adult_income': adult_preprocessor
    }

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
    #param_df['seed'] = np.random.randint(999999, size=num_tests)

    results = None
    with multiprocessing.Pool(NUM_TASKS) as p:
        results = p.map(test_launcher, params)

    param_df = pd.DataFrame(params).drop(['dataset_payload', 'preprocessor_payload', 'model_payload'], axis=1)
    write_dataframe(param_df, results, output_file)
    print("Finished experiment.")


if __name__ == '__main__':
    run_experiment()
