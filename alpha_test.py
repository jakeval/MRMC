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


SCRATCH_DIR = '/mnt/nfs/scratch1/jasonvallada'
OUTPUT_DIR = '/home/jasonvallada/alpha_output'
LOG_DIR = '/home/jasonvallada/MRMC/logs'

def test_launcher(keys, params):
    p = dict([(key, val) for key, val in zip(keys, params)])
    dataset, preprocessor = None, None
    model = model_utils.load_model(p['model'], p['dataset'])
    if p['dataset'] == 'adult_income':
        dataset, _, preprocessor = da.load_adult_income_dataset()
    elif p['dataset'] == 'german_credit':
        dataset, _, preprocessor = da.load_german_credit_dataset()

    X = np.array(preprocessor.transform(dataset.drop('Y', axis=1)))
    model_scores = model.predict_proba(X)
    dataset = da.filter_from_model(dataset, model_scores)

    num_trials = p['num_trials']
    k_dirs = p['k_dirs']
    max_iterations = p['max_iterations']
    experiment_immutable_feature_names = p['experiment_immutable_features']
    validate = p['validate']
    early_stopping = None

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

    mrm = MRM(alpha=alpha_function, weight_function=weight_function, perturb_dir=None)
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
        'Positive Probability': lambda poi, paths: point_stats.check_positive_probability(model, poi, paths),
        'Point Invalidity': lambda poi, points: point_stats.check_validity(preprocessor, column_names_per_feature, poi, points),
        'Final Point Distance': point_stats.check_final_point_distance,
        'Point Immutable Violations': lambda poi, points: point_stats.check_immutability(preprocessor, experiment_immutable_feature_names, poi, points),
        'Point Sparsity': lambda poi, points: point_stats.check_sparsity(preprocessor, poi, points),
    }
    cluster_statistics = {
        'Cluster Size': path_stats.check_cluster_size,
    }

    test = MrmcTestRunner(num_trials, dataset, preprocessor, mrmc, path_statistics,
                        point_statistics, cluster_statistics)
    stats, aggregated_stats = test.run_test()
    return aggregated_stats


def get_params(num_trials):
    """Create a dataframe of input parameters.
    
    Each row of the dataframe contains a setting over parameters used by test_launcher to launch a test.
    Grid search is performed over these parameters, testing every combination of values for all the parameters.
    Simple parameters can be searched over naively. Parameters like early stopping are handled more carefully --
    if early stopping is disabled (early_stopping = False), there is no point in searching over early_stopping_cutoff.
    """

    # simple parameters have no conflicts
    simple_params = {
        'num_trials': [num_trials],
        'k_dirs': [1,2,4],
        'max_iterations': [15],
        'validate': [False],
        'model': ['random_forest', 'svc'],
        'perturb_dir': [None]
    }

    perturb_dir = [
        {
            'perturb_dir': ['random'],
            'perturb_dir_random_scale': [0.01, 0.1, 1]
        },
        {
            'perturb_dir': [None]
        }
    ]

    # the 'model' and 'experiment_immutable_features' parameter values depend on the 'dataset' value
    dataset = [
        {
            'dataset': ['adult_income'],
            'experiment_immutable_features': [['age', 'sex', 'race']],
        },
        {
            'dataset': ['german_credit'],
            'experiment_immutable_features': [['age', 'sex']],
        }
    ]

    early_stopping = [
        {
            'early_stopping': [True],
            'early_stopping_cutoff': [0.6, 0.75, 0.9],
        },
        {
            'early_stopping': [False],
        },
    ]

    weight_function = [
        {
            'weight_function': ['centroid'],
            'weight_centroid_alpha': [0.3, 0.5, 0.7]
        },
        {
            'weight_function': ['constant'],
            'weight_constant_size': [0.5, 1, 1.5]
        }
    ]

    alpha_function = [
        {
            'alpha_function': ['volcano'],
            'alpha_volcano_cutoff': [0.2, 0.5, 0.8],
            'alpha_volcano_degree': [2, 4, 8, 16]
        },
        {
            'alpha_function': ['normal'],
            'alpha_normal_width': [0.5, 1, 2]
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
    output_file = os.path.join(OUTPUT_DIR, 'test_results_3.pkl')

    print("Open a client...")
    cluster = SLURMCluster(
        processes=1,
        memory='1000MB',
        queue='defq',
        cores=1,
        walltime='04:00:00',
        log_directory=LOG_DIR
    )
    cluster.scale(72)
    dask.config.set(scheduler='processes')
    dask.config.set({'temporary-directory': SCRATCH_DIR})
    client = Client(cluster)

    all_params = get_params(30)
    num_tests = all_params.shape[0]
    if len(args) > 1:
        num_tests = int(args[1])
    print(f"Run {num_tests} tests")
    param_df = all_params.iloc[0:num_tests]
    run_test = lambda params: test_launcher(list(param_df.columns), params)
    
    futures = client.map(run_test, param_df.values)
    results = client.gather(futures)
    write_dataframe(param_df, results, output_file)
    print("Finished experiment.")


if __name__ == '__main__':
    run_experiment()
