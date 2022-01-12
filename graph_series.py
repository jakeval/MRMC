
from core.mrmc import MRM, MRMCIterator, MRMIterator
from core import utils
from data import data_adapter as da
from models import model_utils
from face import core
from sklearn.neighbors import KernelDensity
import pandas as pd

from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import sys

RUN_LOCALLY = False
dir = '/home/jasonvallada/MRMC/face_graph'
LOG_DIR = '/home/jasonvallada/MRMC/logs'
SCRATCH_DIR = '/mnt/nfs/scratch1/jasonvallada'
if RUN_LOCALLY:
    dir = './face_graph'
    LOG_DIR = '.'
    SCRATCH_DIR = '.'

def score_dataset(bandwidth, dataset, distance_threshold, conditions, num_trials):
    block_size = None #80000000
    print(f"bandwidth {bandwidth} \ndataset {dataset} \ndistance {distance_threshold} \nconditions {conditions}")
    print("Load the dataset...")
    data = None
    preprocessor = None
    immutable_features = None
    if dataset == 'adult_income':
        data, _, preprocessor = da.load_adult_income_dataset()
        immutable_features = ['age', 'sex', 'race']
    else:
        data, _, preprocessor = da.load_german_credit_dataset()
        immutable_features = ['age', 'sex']
    if num_trials == 0:
        num_trials = data.shape[0]
    random_idx = np.random.choice(np.arange(data.shape[0]), num_trials, replace=False)
    data = data.iloc[random_idx,:]
    
    k_paths = 4
    confidence_threshold = 0.75
    density_threshold = 0.01

    model = model_utils.load_model('random_forest', dataset)
    clf = lambda X: model.predict_proba(X)[:,1]
    conditions_function = None
    if conditions:
        X = preprocessor.transform(data)
        immutable_columns = preprocessor.get_feature_names_out(immutable_features)
        tolerances = None
        immutable_column_indices = np.arange(X.columns.shape[0])[X.columns.isin(immutable_columns)]
        if 'age' in immutable_features:
            age_index = np.arange(X.columns.shape[0])[X.columns == 'age'][0]
            immutable_column_indices = immutable_column_indices[immutable_column_indices != age_index]
            transformed_unit = preprocessor.sc_dict['age'].transform([[1]])[0] - preprocessor.sc_dict['age'].transform([[0]])[0]
            tolerances = {
                age_index: transformed_unit * 5.5
            }
        conditions_function = lambda differences: core.immutable_conditions(differences, immutable_column_indices, tolerances=tolerances)

    face = core.Face(k_paths, clf, distance_threshold, confidence_threshold, density_threshold, conditions_function=conditions_function)
    face.set_graph(preprocessor, data, dataset, bandwidth, bandwidth=bandwidth, dir=dir)
    face.fit(data, preprocessor, verbose=True)
    paths = face.iterate(0)
    print("Finished! Paths are...")
    for i, path in enumerate(paths):
        print(f"Path {i}")
        print(preprocessor.inverse_transform(path))


if __name__ == '__main__':
    args = sys.argv
    bandwidth = float(args[1])
    distance = float(args[2])
    conditions = bool(int(args[3]))
    dataset = args[4]
    num_trials = int(args[5])
    score_dataset(bandwidth, dataset, distance, conditions, num_trials)
