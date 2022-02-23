import numpy as np
from data import data_adapter as da
from experiments.test_face_path import FacePathTestRunner
from face.core import Face, immutable_conditions
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

NUM_TASKS = 48

RUN_LOCALLY = False
INPUT_DIR = '/mnt/nfs/home/jasonvallada/face_graphs'
OUTPUT_DIR = '/mnt/nfs/home/jasonvallada/face_path_output'
if RUN_LOCALLY:
    INPUT_DIR = './face_graphs'
    OUTPUT_DIR = '../face_path_output'
    NUM_TASKS = 1

def test_launcher(p):
    if RUN_LOCALLY:
        print(f"Begin Test {p['distance_threshold']}")
    model = p['model_payload']
    clf = lambda X: model.predict_proba(X)[:,1]
    preprocessor = p['preprocessor_payload']
    dataset = p['dataset_payload']
    graph = p['graph_payload']
    density_scores = p['density_payload']

    np.random.seed(p['seed'])
    poi_index = p['poi_index']
    poi = dataset.loc[[poi_index]].drop('Y', axis=1)

    get_positive_probability = lambda p: model.predict_proba(p.to_numpy())[0,1]

    num_trials = p['num_trials']
    k_dirs = p['k_dirs']
    experiment_immutable_feature_names = None
    if p['dataset'] == 'adult_income':
        experiment_immutable_feature_names = ['age', 'race', 'sex']
    elif p['dataset'] == 'german_credit':
        experiment_immutable_feature_names = ['age', 'sex']

    column_names_per_feature = [] # list of lists
    for feature in dataset.columns.difference(['Y']):
        column_names_per_feature.append(preprocessor.get_feature_names_out([feature]))

    path_statistics = {
        'Path Count': path_stats.check_path_count,
        'Path Length': path_stats.check_path_length,
        'Path Immutable Violations': lambda paths: path_stats.check_immutability(preprocessor, experiment_immutable_feature_names, paths),
        'Average Path Sparsity': lambda paths: path_stats.check_sparsity(preprocessor, paths),
        'Path Invalidity': lambda paths: path_stats.check_validity(preprocessor, column_names_per_feature, paths),
    }
    point_statistics = {
        'Positive Probability': lambda poi, paths: point_stats.check_positive_probability(model, poi, paths, p['confidence_threshold']),
        'Point Invalidity': lambda poi, points: point_stats.check_validity(preprocessor, column_names_per_feature, poi, points),
        'Final Point Distance': point_stats.check_final_point_distance,
        'Point Immutable Violations': lambda poi, points: point_stats.check_immutability(preprocessor, experiment_immutable_feature_names, poi, points),
        'Point Sparsity': lambda poi, points: point_stats.check_sparsity(preprocessor, poi, points),
        'Diversity': point_stats.check_diversity
    }

    distance_threshold = p['distance_threshold']
    density_threshold = p['density_threshold']
    confidence_threshold = p['confidence_threshold']
    immutable_features = None
    feature_tolerances = None
    if p['immutable_features']:
        immutable_features = experiment_immutable_feature_names
        feature_tolerances = {
            'age': 5.1
        }

    #weight_function = lambda dir: utils.scale_weight_function(dir, p['weight_function_alpha'])
    weight_function = lambda dir: utils.constant_step_size(dir, p['step_size'])

    perturb_dir = None
    if p['perturb_dir_random_scale'] > 0:
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

    max_iterations = p['max_iterations']

    kde_bandwidth = p['kde_bandwidth']
    kde_rtol = None if p['kde_rtol'] == 0 else p['kde_rtol']

    face = Face(k_dirs,
                clf,
                distance_threshold,
                confidence_threshold,
                density_threshold)
    face.set_graph_from_memory(graph, density_scores, kde_bandwidth, kde_rtol)
    test = FacePathTestRunner(num_trials,
                              dataset,
                              preprocessor,
                              face,
                              point_statistics,
                              path_statistics,
                              k_dirs,
                              weight_function,
                              get_positive_probability,
                              perturb_dir,
                              max_iterations,
                              None,
                              immutable_features=immutable_features,
                              feature_tolerances=feature_tolerances)
    stats, paths = test.run_trial(poi)
    paths_indexed = []
    for path in paths:
        path = path.copy()
        path['path_order'] = np.arange(path.shape[0])
        paths_indexed.append(path)
    return paths_indexed


def scale_weight_function(dir, immutable_column_indices, rescale_factor):
    new_dir = dir * rescale_factor
    new_dir[:,immutable_column_indices] = dir[:,immutable_column_indices]
    return new_dir


def get_params(dataset_str, dataset_poi_indices, seed):
    """Create a dataframe of input parameters.
    
    Each row of the dataframe contains a setting over parameters used by test_launcher to launch a test.
    Grid search is performed over these parameters, testing every combination of values for all the parameters.
    Simple parameters can be searched over naively. Parameters like early stopping are handled more carefully --
    if early stopping is disabled (early_stopping = False), there is no point in searching over early_stopping_cutoff.
    """

    # simple parameters have no conflicts
    simple_params = {
        'seed': [seed],
        'num_trials': [1],
        'k_dirs': [4],
        'max_iterations': [15],
        'model': ['svc', 'random_forest'],
        'confidence_threshold': [0.7],
        'perturb_dir_random_scale': [0, 0.25, 0.5, 0.75, 1],
        'step_size': [1,1.25,1.5],
        'poi_index': dataset_poi_indices
    }

    dataset = None
    if dataset_str == 'adult_income':
        dataset = [
            {
            'dataset': ['adult_income'],
            'immutable_features': [True],
            'kde_bandwidth': [0.13],
            'kde_rtol': [1000],
            'density_threshold': [0],
            'distance_threshold': [2],
            }
        ]
    elif dataset_str == 'german_credit':
        dataset = [
            {
            'dataset': ['german_credit'],
            'immutable_features': [True],
            'kde_bandwidth': [0.29],
            'kde_rtol': [0],
            'density_threshold': [0], # anything below this number will be culled
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


def clean_dict(d):
    d2 = {}
    for key, val in d.items():
        if key not in ['dataset_payload', 'preprocessor_payload', 'model_payload', 'density_payload', 'graph_payload']:
            d2[key] = val
    return d2


def write_dataframe(columns, preprocessor, param_dicts, results_list, output_file):
    results_dataframes = []
    trial_key = 0
    for param_dict, results in zip(param_dicts, results_list):
        paths = results
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


def aux_data_from_params(params):
    cache = {}
    density_score_list = []
    graph_list = []
    for param_dict in params:
        dataset = param_dict['dataset']
        bandwidth = param_dict['kde_bandwidth']
        rtol = param_dict['kde_rtol']
        if rtol == 0:
            rtol = None
        distance_threshold = param_dict['distance_threshold']
        use_conditions = param_dict['immutable_features']
        awkward_key = f"{dataset}-{bandwidth}-{rtol}-{distance_threshold}-{use_conditions}"
        if awkward_key not in cache:
            cache[awkward_key] = Face.load_graph(dataset, bandwidth, distance_threshold, False, rtol=rtol, dir=INPUT_DIR)
        density_scores, graph = cache[awkward_key]
        density_score_list.append(density_scores)
        graph_list.append(graph)
    return density_score_list, graph_list


def run_experiment():
    print("starting the script...")
    args = sys.argv
    num_tests = None
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
    density_score_list, graph_list = aux_data_from_params(params)
    for param_dict, density_score, graph in zip(params, density_score_list, graph_list):
        new_params = {
            'dataset_payload': dataset_payload,
            'preprocessor_payload': preprocessor_payload,
            'model_payload': models[(param_dict['model'], param_dict['dataset'])],
            'density_payload': density_score,
            'graph_payload': graph
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
