import numpy as np
from data import data_adapter as da
from models import random_forest
from experiments.test_mrmc import MrmcTestRunner
from core.mrmc import MRM, MRMCIterator, MRMIterator
from experiments import path_stats, point_stats
from core import utils
import dask
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import pandas as pd
from sklearn.model_selection import ParameterGrid
import itertools
import sys
from models import model_utils
import os

np.random.seed(885577)

SCRATCH_DIR = '/mnt/nfs/scratch1/jasonvallada'
OUTPUT_DIR = '/home/jasonvallada/alpha_output'
LOG_DIR = '/home/jasonvallada/MRMC/logs'

def test_launcher(models, preprocessors, keys, params, dataset):
    p = dict([(key, val) for key, val in zip(keys, params)])
    model = models[(p['model'], p['dataset'])]
    preprocessor = preprocessors[p['dataset']]

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
    if p['immutable_features'] is not None:
        immutable_column_names = preprocessor.get_feature_names_out(p['immutable_features'])
        immutable_features = p['immutable_features']
        feature_tolerances = {
            'age': 5
        }

    perturb_dir = None
    if p['perturb_dir'] == 'random':
        perturb_dir = lambda dir: utils.random_perturb_dir(p['perturb_dir_random_scale'])

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
                        point_statistics, cluster_statistics, immutable_features=immutable_features,
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
        'validate': [False],
        'model': ['random_forest', 'svc'],
        'perturb_dir': [None]
    }

    dataset = []
    if dataset_str == 'adult_income':
        dataset.append(
            {
                'dataset': ['adult_income'],
                'experiment_immutable_features': [['age', 'sex', 'race']],
                'immutable_features': [['age', 'sex', 'race'], None],
            })
    else:
        dataset.append(
            {
                'dataset': ['german_credit'],
                'experiment_immutable_features': [['age', 'sex']],
                'immutable_features': [['age', 'sex'], None],
            })

    early_stopping = [
        {
            'early_stopping': [True],
            'early_stopping_cutoff': [0.6, 0.7, 0.8],
        }
    ]

    weight_function = [
        {
            'weight_function': ['centroid'],
            'weight_centroid_alpha': [0.3, 0.5, 0.7]
        },
    ]

    alpha_function = [
        {
            'alpha_function': ['volcano'],
            'alpha_volcano_cutoff': [0.2, 0.5, 0.8],
            'alpha_volcano_degree': [2, 4, 8]
        },
        {
            'alpha_function': ['normal'],
            'alpha_normal_width': [0.5, 1, 2, 4]
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
    
    params = pd.DataFrame(ParameterGrid(params))
    print("param count: ")
    print(params.shape[0])
    return params


def write_dataframe(params_df, results_dataframe_list, output_file):
    results_dataframe = pd.concat(results_dataframe_list, axis=0).reset_index()
    results_dataframe = results_dataframe
    final_df = pd.concat([params_df, results_dataframe], axis=1)
    final_df.to_pickle(output_file)


def run_experiment():
    print("starting the script...")
    args = sys.argv
    num_tests = 0
    dataset = None
    if len(args) > 1:
        dataset = args[1]
        num_tests = int(args[2])
    output_file = os.path.join(OUTPUT_DIR, f'{dataset}.pkl')

    print("Open a client...")
    cluster = SLURMCluster(
        processes=1,
        memory='1000MB',
        queue='defq',
        cores=1,
        walltime='00:20:00',
        log_directory=LOG_DIR
    )
    cluster.scale(72)
    dask.config.set(scheduler='processes')
    dask.config.set({'temporary-directory': SCRATCH_DIR})
    client = Client(cluster)
    #client = Client(n_workers=1, threads_per_worker=1)

    models = {
        ('svc', 'german_credit'): model_utils.load_model('svc', 'german_credit'),
        ('svc', 'adult_income'): model_utils.load_model('svc', 'adult_income'),
        ('random_forest', 'german_credit'): model_utils.load_model('random_forest', 'german_credit'),
        ('random_forest', 'adult_income'): model_utils.load_model('random_forest', 'adult_income'),
    }

    german_data, _, german_preprocessor = da.load_german_credit_dataset()
    adult_data, _, adult_preprocessor = da.load_adult_income_dataset()

    german_future = client.scatter([german_data], broadcast=True)[0]
    adult_future = client.scatter([adult_data], broadcast=True)[0]

    preprocessors = {
        'german_credit': german_preprocessor,
        'adult_income': adult_preprocessor
    }

    all_params = get_params(30, dataset)
    if num_tests == 0:
        num_tests = all_params.shape[0]
    print(f"Run {num_tests} tests")
    param_df = all_params.iloc[0:num_tests]
    run_test = lambda params, dataset: test_launcher(models, preprocessors, list(param_df.columns), params, dataset)
    params = param_df.values
    dataset_futures = list(map(lambda dataset: german_future if dataset == 'german_credit' else adult_future, param_df.loc[:,'dataset']))
    futures = client.map(run_test, params, dataset_futures)
    results = client.gather(futures)
    write_dataframe(param_df, results, output_file)
    print("Finished experiment.")


if __name__ == '__main__':
    run_experiment()
