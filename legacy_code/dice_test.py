import numpy as np
from data import data_adapter as da
from experiments.test_dice import DiceTestRunner
from experiments import point_stats
from models import model_utils
import dice_ml
from sklearn.pipeline import Pipeline

import dask
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import pandas as pd
from sklearn.model_selection import ParameterGrid
import sys
import os

np.random.seed(88557)

RUN_LOCALLY = True
SCRATCH_DIR = '/mnt/nfs/scratch1/jasonvallada'
OUTPUT_DIR = '/home/jasonvallada/dice_output'
LOG_DIR = '/home/jasonvallada/MRMC/logs'
if RUN_LOCALLY:
    SCRATCH_DIR = '.'
    OUTPUT_DIR = '../dice_output'
    LOG_DIR = '.'

class ToNumpy:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def transform(self, X):
        return X.to_numpy()

def test_launcher(models, preprocessors, keys, params, dataset):
    p = dict([(key, val) for key, val in zip(keys, params)])
    if RUN_LOCALLY:
        print("Begin Test")
        print(p)
    model = models[(p['model'], p['dataset'])]
    preprocessor = preprocessors[p['dataset']]

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
        

    point_statistics = {
        'Positive Probability': lambda poi, paths: point_stats.check_positive_probability(model, poi, paths, p['certainty_cutoff']),
        'Point Invalidity': lambda poi, points: point_stats.check_validity(preprocessor, column_names_per_feature, poi, points),
        'Final Point Distance': point_stats.check_final_point_distance,
        'Point Immutable Violations': lambda poi, points: point_stats.check_immutability(preprocessor, experiment_immutable_feature_names, poi, points),
        'Point Sparsity': lambda poi, points: point_stats.check_sparsity(preprocessor, poi, points),
        'Diversity': point_stats.check_diversity,
        'Model Certainty': lambda poi, cf_points: point_stats.check_model_certainty(model, poi, cf_points)
    }

    k_points = p['k_points']
    immutable_features = None
    feature_tolerances = None
    if p['immutable_features']:
        if p['dataset'] == 'adult_income':
            immutable_features = ['age', 'race', 'sex']
        if p['dataset'] == 'german_credit':
            immutable_features = ['age', 'sex']
        feature_tolerances = {'age': 5}

    num_trials = p['num_trials']
    certainty_cutoff = p['certainty_cutoff']
    test = DiceTestRunner(num_trials, dataset, preprocessor, 
                        dice, point_statistics, k_points,
                        certainty_cutoff,
                        immutable_features=immutable_features,
                        feature_tolerances=feature_tolerances)

    _, aggregated_stats = test.run_test()
    return aggregated_stats


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
        'k_points': [4],
        'model': ['svc', 'random_forest'],
        'immutable_features': [True],
        'dataset': [dataset_str],
        'certainty_cutoff': [0.6, 0.7]
    }
    
    params = pd.DataFrame(ParameterGrid(params))
    print("param count: ")
    print(params.shape[0])
    return params


def write_dataframe(params_df, results_dataframe_list, output_file):
    results_dataframe = pd.concat(results_dataframe_list, axis=0).reset_index()
    results_dataframe = results_dataframe
    final_df = pd.concat([params_df.reset_index(), results_dataframe], axis=1)
    final_df.to_pickle(output_file)
    if RUN_LOCALLY:
        outputs = [
            'dataset',
            'model',
            'certainty_cutoff',
            'Positive Probability (mean)',
            #'Model Certainty (mean)',
            'Final Point Distance (mean)',
            #'Point Sparsity (mean)',
            'Point Immutable Violations (mean)',
            'Diversity (mean)'
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

    print("Open a client...")
    client = None
    num_trials = 30
    if not RUN_LOCALLY:
        dask.config.set(scheduler='processes')
        dask.config.set({'temporary-directory': SCRATCH_DIR})
        cluster = SLURMCluster(
            processes=1,
            memory='2000MB',
            queue='defq',
            cores=1,
            walltime='00:25:00',
            log_directory=LOG_DIR
        )
        cluster.scale(2)
        client = Client(cluster)
    else:
        client = Client(n_workers=1, threads_per_worker=1)

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
