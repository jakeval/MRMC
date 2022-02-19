import numpy as np
from data import data_adapter as da
# from experiments.test_dice_path import DicePathTestRunner
from experiments.test_mrmc import MrmcTestRunner
from experiments import point_stats, path_stats
from models import model_utils
from core import utils
from sklearn.pipeline import Pipeline
from core.mrmc import MRM, MRMCIterator

from matplotlib import pyplot as plt
from visualize.two_d_plots import Display2DPaths

import multiprocessing
import pandas as pd
from sklearn.model_selection import ParameterGrid
import sys
import os

sys.path.append(os.getcwd())
np.random.seed(88557)

RUN_LOCALLY = True
RUN_ALL = True
NUM_TASKS = 12
OUTPUT_DIR = '/mnt/nfs/home/jasonvallada/dice_path_output'
if RUN_LOCALLY:
    plt.style.use('style.mplstyle')
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

    np.random.seed(p['poi_seed'])
    poi_index = np.random.choice(dataset[dataset.Y == -1].index, 1)[0]
    poi_index = 10308
    poi = dataset.loc[dataset.index == poi_index].drop('Y', axis=1)
    print("poi: ", poi_index)
    seed = 885577 # p['seed']
    np.random.seed(seed)

    X = np.array(preprocessor.transform(dataset.drop('Y', axis=1)))
    model_scores = model.predict_proba(X)
    dataset = da.filter_from_model(dataset, model_scores)

    column_names_per_feature = [] # list of lists
    for feature in dataset.columns.difference(['Y']):
        column_names_per_feature.append(preprocessor.get_feature_names_out([feature]))

    experiment_immutable_feature_names = None
    if p['dataset'] == 'adult_income':
        experiment_immutable_feature_names = ['age', 'race', 'sex']
    elif p['dataset'] == 'german_credit':
        experiment_immutable_feature_names = ['age', 'sex']


    num_trials = p['num_trials']
    k_dirs = p['k_dirs']
    max_iterations = p['max_iterations']
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

    if p['early_stopping'] is not None:
        early_stopping = lambda point: utils.model_early_stopping(model, point, cutoff=p['early_stopping'])
    weight_function = None
    if p['weight_function'] == 'centroid':
        weight_function = lambda dir, poi, X: utils.constant_step_size(dir, 1.5)
        # weight_function = lambda dir, poi, X: utils.centroid_normalization(dir, poi, X, alpha=p['weight_centroid_alpha'])
    alpha_function = None
    if p['alpha_function'] == 'volcano':
        alpha_function = lambda dist: utils.volcano_alpha(dist, cutoff=p['alpha_volcano_cutoff'], degree=p['alpha_volcano_degree'])
    elif p['alpha_function'] == 'normal':
        alpha_function = lambda dist: utils.normal_alpha(dist, width=p['alpha_normal_width'])

    mrm = MRM(alpha=alpha_function, weight_function=weight_function, perturb_dir=perturb_dir, immutable_column_names=immutable_column_names)
    mrmc = MRMCIterator(k_dirs, mrm, preprocessor, max_iterations, early_stopping=early_stopping, validate=validate)

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
    cluster_statistics = {
        'Cluster Size': path_stats.check_cluster_size,
    }

    test = MrmcTestRunner(
        num_trials,
        dataset,
        preprocessor,
        mrmc,
        path_statistics,
        point_statistics,
        cluster_statistics,
        None,
        immutable_features=immutable_features,
        immutable_strict=False,
        feature_tolerances=feature_tolerances,
        check_privacy=False)

    stats, paths, cluster_centers = test.run_trial(poi)
    #print("GOT RESULTS")
    #print(len(paths))
    for path, pp in zip(paths, stats['Positive Probability']):
        print(f"path ({pp}):")
        #print(path.iloc[0,:])
        #print(path.iloc[1,:])
        print(np.linalg.norm(path.iloc[0,:] - path.iloc[1,:]))
        # print(preprocessor.inverse_transform(path))
    return paths, cluster_centers, p['perturb_dir_random_scale']

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
        'max_iterations': [15],
        'perturb_dir_random_scale': [0,.25,0.5,0.75,1],
        'validate': [True],
        'sparsity': [False],
        'early_stopping': [0.7],
        'alpha_function': ['volcano'],
        'alpha_volcano_degree': [1],
        'alpha_volcano_cutoff': [0.2],
        'weight_function': ['centroid'],
        'weight_centroid_alpha': [0.7],
        'k_dirs': [4],
        'model': ['random_forest'],
        'immutable_features': [True],
        'dataset': [dataset_str],
        'certainty_cutoff': [0.7]
    }
    if not RUN_ALL:
        return [{
            'num_trials': num_trials,
            'max_iterations': 15,
            'perturb_dir_random_scale': 4,
            'validate': True,
            'sparsity': True,
            'early_stopping': 0.7,
            'alpha_function': 'normal',
            'alpha_normal_width': 0.5,
            'weight_function': 'centroid',
            'weight_centroid_alpha': 0.3,
            'k_dirs': 4,
            'model': 'random_forest',
            'immutable_features': True,
            'dataset': dataset_str,
            'certainty_cutoff': 0.7
        }]
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


def visualize_paths(path_sets, X, Y):
    cluster_centers = None
    for paths, cluster_centers, random_scale in path_sets:
        cluster_centers = cluster_centers
        display = Display2DPaths(X, Y)
        fig, ax = display.set_paths(paths).set_clusters(cluster_centers).do_pca().scatter()
        ax.set_title(f"Scale {random_scale}")
        ax.legend()
    plt.show()


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

    data, preprocessor = None, None
    if dataset == 'adult_income':
        data, _, preprocessor = da.load_adult_income_dataset()
    elif dataset == 'german_credit':
        data, _, preprocessor = da.load_german_credit_dataset()
    else:
        print('no dataset of that name')
        return
    
    poi_seed = 148295 #148294

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
            'dataset_payload': data,
            'preprocessor_payload': preprocessor,
            'model_payload': models[(param_dict['model'], param_dict['dataset'])]
        }
        param_dict.update(new_params)

    path_sets = None
    with multiprocessing.Pool(NUM_TASKS) as p:
        path_sets = p.map(test_launcher, params)

    X = preprocessor.transform(data.drop('Y', axis=1)).to_numpy()
    Y = data.Y.to_numpy()
    visualize_paths(path_sets, X, Y)
    # param_df = pd.DataFrame(params).drop(['dataset_payload', 'preprocessor_payload', 'model_payload'], axis=1)

    # write_dataframe(param_df, results, output_file)
    # print("Finished experiment.")


if __name__ == '__main__':
    run_experiment()
