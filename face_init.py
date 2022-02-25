import numpy as np
from data import data_adapter as da
from face.core import Face, immutable_conditions
import pandas as pd
from sklearn.model_selection import ParameterGrid
import itertools
import sys
import os
import multiprocessing

sys.path.append(os.getcwd())
np.random.seed(88557)

NUM_TASKS = 1

RUN_LOCALLY = False
OUTPUT_DIR = '/mnt/nfs/home/jasonvallada/face_graphs'
if RUN_LOCALLY:
    NUM_TASKS = 1
    OUTPUT_DIR = './face_graphs'


def get_params(dataset_str):
    """Create a list of dictionaries of input parameters.
    
    Each dictionary contains a setting over parameters used by test_launcher to launch a test.
    Grid search is performed over these parameters, testing every combination of values for all the parameters.
    Simple parameters can be searched over naively. Parameters like early stopping are handled more carefully --
    if early stopping is disabled (early_stopping = False), there is no point in searching over early_stopping_cutoff.
    """

    # simple parameters have no conflicts
    simple_params = {}

    dataset = None
    if dataset_str == 'adult_income':
        dataset = [
            {
            'dataset': ['adult_income'],
            'immutable_features': [None],
            'kde_bandwidth': [0.13],
            'kde_rtol': [1000],
            'distance_threshold': [2],
            }
        ]
    elif dataset_str == 'german_credit':
        dataset = [
            {
            'dataset': ['german_credit'],
            'immutable_features': [None],
            'kde_bandwidth': [0.29],
            'kde_rtol': [None],
            'distance_threshold': [8],
            }
        ]

    # the simple and constrained parameters are combined into a list of dictionaries which respect the parameter constraints
    constrained_params = [dataset]
    params = []
    for dict_tuple in itertools.product(*constrained_params):
        d = {}
        for dict in dict_tuple:
            d.update(dict)
        d.update(simple_params)
        params.append(d)
    return list(ParameterGrid(params))

def generate_density_scores(dataset, preprocessor, dataset_str):
    bandwidth = None
    rtol = None
    if dataset_str == 'adult_income':
        bandwidth = 0.13
        rtol = 1000
    else:
        bandwidth = 0.29
        rtol = None
    return Face.generate_kde_scores(preprocessor, dataset, bandwidth, rtol=rtol, save_results=True, dataset_str=dataset_str, dir=OUTPUT_DIR)
 
def generate_graph(p):
    np.random.seed(p['seed'])
    Face.generate_graph(p['preprocessor_payload'],
                        p['dataset_payload'],
                        p['kde_bandwidth'],
                        p['density_scores_payload'],
                        p['distance_threshold'],
                        rtol=p['kde_rtol'],
                        dir=OUTPUT_DIR,
                        save_results=True,
                        dataset_str=p['dataset'])


def main():
    args = sys.argv
    num_tests = int(args[2])
    dataset = args[1]
    print("dataset is ", dataset)
    data, preprocessor = None, None
    if dataset == 'adult_income':
        data, _, preprocessor = da.load_adult_income_dataset()
    elif dataset == 'german_credit':
        data, _, preprocessor = da.load_german_credit_dataset()
    else:
        print("No dataset loaded")
        return
    print("Generate KDE scores...")
    # density_scores = generate_density_scores(data, preprocessor, dataset)
    density_scores = Face.load_kde('adult_income', 0.13, rtol=1000, dir=OUTPUT_DIR)
    print("Generated density scores")

    all_params = get_params(dataset)
    print(len(all_params))
    if num_tests == 0:
        num_tests = len(all_params)
    print(f"Run {num_tests} tests")
    params = all_params[:num_tests]
    for param_dict in params:
        new_params = {
            'seed': np.random.randint(999999),
            'dataset_payload': data,
            'preprocessor_payload': preprocessor,
            'density_scores_payload': density_scores
        }
        param_dict.update(new_params)

    with multiprocessing.Pool(NUM_TASKS) as p:
        p.map(generate_graph, params)

    print("Finished!")


if __name__ == "__main__":
    print("starting the script...")
    main()