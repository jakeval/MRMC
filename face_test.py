import numpy as np
from data import data_adapter as da
from models import random_forest
from experiments.test_face import FaceTestRunner
from face.core import Face, immutable_conditions
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

np.random.seed(88557)

RUN_LOCALLY = True
SCRATCH_DIR = '/mnt/nfs/scratch1/jasonvallada'
INPUT_DIR = '/home/jasonvallada/MRMC/face_graph'
OUTPUT_DIR = '/home/jasonvallada/face_output'
LOG_DIR = '/home/jasonvallada/MRMC/logs'
if RUN_LOCALLY:
    SCRATCH_DIR = '.'
    OUTPUT_DIR = '.'
    INPUT_DIR = './face_graph'
    LOG_DIR = '.'
# models, preprocessors, list(param_df.columns), params, dataset, density_scores, graph
def test_launcher(models, preprocessors, keys, params, dataset, density_scores, graph):
    p = dict([(key, val) for key, val in zip(keys, params)])
    if RUN_LOCALLY:
        print("Begin Test")
        print(p)
    model = models[(p['model'], p['dataset'])]
    clf = lambda X: model.predict_proba(X)[:,1]
    preprocessor = preprocessors[p['dataset']]

    X = np.array(preprocessor.transform(dataset.drop('Y', axis=1)))

    num_trials = p['num_trials']
    k_dirs = p['k_dirs']
    experiment_immutable_feature_names = p['experiment_immutable_features']
    immutable_column_names = None
    immutable_features = None
    feature_tolerances = None
    immutable_strict = True
    if p['immutable_features'] is not None:
        immutable_column_names = preprocessor.get_feature_names_out(p['immutable_features'])
        immutable_features = p['immutable_features']
        feature_tolerances = {
            'age': 5
        }



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

    distance_threshold = p['distance_threshold']
    density_threshold = p['density_threshold']
    confidence_threshold = p['confidence_threshold']
    conditions_function = None
    age_tolerance = None
    immutable_features = None
    if p['immutable_features'] is not None:
        immutable_features = p['immutable_features']
        print("use conditions...")
        X = preprocessor.transform(dataset)
        immutable_columns = preprocessor.get_feature_names_out(immutable_features)
        tolerances = None
        immutable_column_indices = np.arange(X.columns.shape[0])[X.columns.isin(immutable_columns)]
        if 'age' in immutable_features:
            age_index = np.arange(X.columns.shape[0])[X.columns == 'age'][0]
            immutable_column_indices = immutable_column_indices[immutable_column_indices != age_index]
            transformed_unit = preprocessor.sc_dict['age'].transform([[1]])[0,0] - preprocessor.sc_dict['age'].transform([[0]])[0,0]
            age_tolerance = 5.5
            tolerances = {
                age_index: transformed_unit * age_tolerance
            }
        conditions_function = lambda differences: immutable_conditions(differences, immutable_column_indices, tolerances=tolerances)

    
    face = Face(k_dirs,
                clf,
                distance_threshold,
                confidence_threshold,
                density_threshold,
                conditions_function=conditions_function)
    face.set_graph_from_memory(graph, density_scores)

    test = FaceTestRunner(num_trials, dataset, preprocessor, face,
                        path_statistics,
                        point_statistics,
                        immutable_features=immutable_features,
                        age_tolerance=age_tolerance)
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
        'model': ['svc', 'random_forest'],
        'confidence_threshold': [0.75],
        'distance_threshold': [1.5,2,4,8],
    }

    dataset = None
    if dataset_str == 'adult_income':
        dataset = [
            {
            'dataset': ['adult_income'],
            'experiment_immutable_features': [['age', 'sex', 'race']],
            'immutable_features': [['age', 'sex', 'race'], None],
            'kde_bandwidth': [0.13],
            'kde_rtol': [1000],
            'density_threshold': [0, np.exp(6), np.exp(7)],
            }
        ]
    elif dataset_str == 'german_credit':
        dataset = [
            {
            'dataset': ['german_credit'],
            'experiment_immutable_features': [['age', 'sex']],
            'immutable_features': [['age', 'sex'], None],
            'kde_bandwidth': [0.29],
            'kde_rtol': [None],
            'density_threshold': [0, np.exp(12.771), np.exp(12.773)], # anything below this number will be culled
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
    
    params = pd.DataFrame(ParameterGrid(params))
    print("param count: ")
    print(params.shape[0])
    return params


def write_dataframe(params_df, results_dataframe_list, output_file):
    results_dataframe = pd.concat(results_dataframe_list, axis=0).reset_index()
    results_dataframe = results_dataframe
    final_df = pd.concat([params_df.reset_index(), results_dataframe], axis=1)
    final_df.to_pickle(output_file)


def aux_data_from_params(param_df):
    cache = {}
    density_score_list = []
    graph_list = []
    for i in range(param_df.shape[0]):
        params = param_df.iloc[i]
        dataset = params['dataset']
        bandwidth = params['kde_bandwidth']
        rtol = params['kde_rtol']
        distance_threshold = params['distance_threshold']
        use_conditions = params['immutable_features'] is not None
        awkward_key = f"{dataset}-{bandwidth}-{rtol}-{distance_threshold}-{use_conditions}"
        if awkward_key not in cache:
            cache[awkward_key] = Face.load_graph(dataset, bandwidth, distance_threshold, use_conditions, rtol=rtol, dir=INPUT_DIR)
        density_scores, graph = cache[awkward_key]
        density_score_list.append(density_scores)
        graph_list.append(graph)
    return density_score_list, graph_list
        

def run_experiment():
    print("starting the script...")
    args = sys.argv
    num_tests = None
    dataset = None
    if len(args) < 2:
        print("Not enough arguments")
        return
    if len(args) == 2:
        dataset = args[1]
    if len(args) == 3:
        num_tests = int(args[1])
        dataset = args[2]
    print("dataset is ", dataset)
    output_file = os.path.join(OUTPUT_DIR, f'{dataset}.pkl')

    print("Open a client...")
    client = None
    num_trials = 30
    if not RUN_LOCALLY:
        cluster = SLURMCluster(
            processes=1,
            memory='2000MB',
            queue='defq',
            cores=1,
            walltime='00:25:00',
            log_directory=LOG_DIR
        )
        cluster.scale(32)
        client = Client(cluster)
    else:
        num_trials = 5
        client = Client(n_workers=1, threads_per_worker=1)
    dask.config.set(scheduler='processes')
    dask.config.set({'temporary-directory': SCRATCH_DIR})

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

    all_params = get_params(num_trials, dataset)
    print(all_params.shape)
    if num_tests is None:
        num_tests = all_params.shape[0]
    print(f"Run {num_tests} tests")
    param_df = all_params.iloc[0:num_tests]
    run_test = lambda params, dataset, density_scores, graph: test_launcher(models, preprocessors, list(param_df.columns), params, dataset, density_scores, graph)
    params = param_df.values
    dataset_futures = list(map(lambda dataset: german_future if dataset == 'german_credit' else adult_future, param_df.loc[:,'dataset']))
    density_score_list, graph_list = aux_data_from_params(param_df)
    density_futures = list(map(lambda density_scores: client.scatter([density_scores], broadcast=True)[0], density_score_list))
    graph_futures = list(map(lambda graph: client.scatter([graph], broadcast=True)[0], graph_list))
    futures = client.map(run_test, params, dataset_futures, density_futures, graph_futures)
    results = client.gather(futures)
    write_dataframe(param_df, results, output_file)
    print("Finished experiment.")


if __name__ == '__main__':
    run_experiment()
