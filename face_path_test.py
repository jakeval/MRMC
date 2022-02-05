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

NUM_TASKS = 8

RUN_LOCALLY = True
INPUT_DIR = '/mnt/nfs/home/jasonvallada/face_graphs'
OUTPUT_DIR = '/mnt/nfs/home/jasonvallada/face_path_output'
if RUN_LOCALLY:
    INPUT_DIR = './face_graphs'
    OUTPUT_DIR = '../face_path_output'
    NUM_TASKS = 1

def test_launcher(p):
    np.random.seed(p['seed'])
    if RUN_LOCALLY:
        print(f"Begin Test {p['distance_threshold']}")
    model = p['model_payload']
    clf = lambda X: model.predict_proba(X)[:,1]
    preprocessor = p['preprocessor_payload']
    dataset = p['dataset_payload']
    graph = p['graph_payload']
    density_scores = p['density_payload']

    X = np.array(preprocessor.transform(dataset.drop('Y', axis=1)))

    num_trials = p['num_trials']
    k_dirs = p['k_dirs']
    experiment_immutable_feature_names = p['experiment_immutable_features']

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
    conditions_function = None
    age_tolerance = None
    immutable_features = None
    immutable_column_indices = None
    immutable_column_indices_full = None
    if p['immutable_features'] is not None:
        immutable_features = p['immutable_features']
        X = preprocessor.transform(dataset)
        immutable_columns = preprocessor.get_feature_names_out(immutable_features)
        tolerances = None
        immutable_column_indices = np.arange(X.columns.shape[0])[X.columns.isin(immutable_columns)]
        if 'age' in immutable_features:
            age_index = np.arange(X.columns.shape[0])[X.columns == 'age'][0]
            immutable_column_indices_full = immutable_column_indices
            immutable_column_indices = immutable_column_indices[immutable_column_indices != age_index]
            transformed_unit = preprocessor.sc_dict['age'].transform([[1]])[0,0] - preprocessor.sc_dict['age'].transform([[0]])[0,0]
            age_tolerance = 5.5
            tolerances = {
                age_index: transformed_unit * age_tolerance
            }
        conditions_function = lambda differences: immutable_conditions(differences, immutable_column_indices_full, tolerances=tolerances)

    weight_function = lambda dir: scale_weight_function(dir, immutable_column_indices_full, p['weight_function_alpha'])
    perturb_dir = None
    if p['perturb_dir_random_scale'] is not None:
        perturb_dir = lambda dir: utils.random_perturb_dir(p['perturb_dir_random_scale'], dir, immutable_column_indices_full)
    max_iterations = p['max_iterations']

    kde_bandwidth = p['kde_bandwidth']
    kde_rtol = p['kde_rtol']

    face = Face(k_dirs,
                clf,
                distance_threshold,
                confidence_threshold,
                density_threshold,
                conditions_function=conditions_function)
    face.set_graph_from_memory(graph, density_scores, kde_bandwidth, kde_rtol)
    test = FacePathTestRunner(num_trials,
                              dataset,
                              preprocessor,
                              face,
                              point_statistics,
                              path_statistics,
                              k_dirs,
                              weight_function,
                              clf,
                              perturb_dir,
                              max_iterations,
                              immutable_features=immutable_features,
                              age_tolerance=age_tolerance)
    stats, aggregated_stats = test.run_test()
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
    simple_params = {
        'num_trials': [num_trials],
        'k_dirs': [4],
        'max_iterations': [15],
        'model': ['svc', 'random_forest'],
        'confidence_threshold': [0.6, 0.7],
        'distance_threshold': [1,1.25,1.5,2,4],
        'perturb_dir_random_scale': [None, 0.25, 0.5, 1, 2, 4],
        'weight_function_alpha': [0.7]
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
    
    return list(ParameterGrid(params))


def write_dataframe(params_df, results_dataframe_list, output_file):
    results_dataframe = pd.concat(results_dataframe_list, axis=0).reset_index()
    results_dataframe = results_dataframe
    final_df = pd.concat([params_df.reset_index(), results_dataframe], axis=1)
    final_df.to_pickle(output_file)
    if RUN_LOCALLY:
        outputs = [
            'dataset',
            'model',
            'density_threshold',
            'confidence_threshold',
            'Positive Probability (mean)',
            #'Model Certainty (mean)',
            'Final Point Distance (mean)',
            #'Point Sparsity (mean)',
            #'Point Immutable Violations (mean)',
            'success_ratio',
            'Diversity (mean)'
        ]
        print(final_df[outputs])


def aux_data_from_params(params):
    cache = {}
    density_score_list = []
    graph_list = []
    for param_dict in params:
        dataset = param_dict['dataset']
        bandwidth = param_dict['kde_bandwidth']
        rtol = param_dict['kde_rtol']
        distance_threshold = param_dict['distance_threshold']
        use_conditions = param_dict['immutable_features'] is not None
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
    num_tests = int(args[2])
    dataset = args[1]
    print("dataset is ", dataset)
    output_file = os.path.join(OUTPUT_DIR, f'{dataset}.pkl')

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

    all_params = get_params(num_trials, dataset)
    print(len(all_params))
    if num_tests == 0:
        num_tests = len(all_params)
    print(f"Run {num_tests} tests")
    params = all_params[:num_tests]
    density_score_list, graph_list = aux_data_from_params(params)
    for param_dict, density_score, graph in zip(params, density_score_list, graph_list):
        new_params = {
            'seed': np.random.randint(999999),
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
    param_df = pd.DataFrame(params).drop(['dataset_payload', 'preprocessor_payload', 'model_payload', 'density_payload', 'graph_payload'], axis=1)
    write_dataframe(param_df, results, output_file)
    print("Finished experiment.")


if __name__ == '__main__':
    run_experiment()
