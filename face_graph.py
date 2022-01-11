
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
if RUN_LOCALLY:
    dir = './face_graph'

def score_dataset(bandwidth, dataset, distance_threshold, conditions):
    print(f"bandwidth {bandwidth} \ndataset {dataset} \ndistance{distance_threshold} \nconditions {conditions}")
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

    k_paths = 4
    confidence_threshold = 0.7
    density_threshold = 0.01

    clf = None
    conditions_function = None
    if conditions:
        X = preprocessor.transform(data)
        immutable_columns = preprocessor.get_feature_names_out(immutable_features)
        immutable_column_indices = np.arange(X.columns.shape[0])[X.columns.isin(immutable_columns)]
        conditions_function = lambda differences: core.immutable_conditions(differences, immutable_column_indices)

    face = core.Face(k_paths, clf, distance_threshold, confidence_threshold, density_threshold, conditions_function=conditions_function)
    face.set_graph(preprocessor, data, dataset, bandwidth, dir=dir)
    face.fit(data, preprocessor, verbose=True)
    paths = face.iterate(0)
    print("Finished! Paths are...")
    print(paths)


if __name__ == '__main__':
    args = sys.argv
    bandwidth = float(args[1])
    distance = float(args[2])
    conditions = bool(args[3])
    dataset = args[4]
    score_dataset(bandwidth, dataset, distance, conditions)
