
from core.mrmc import MRM, MRMCIterator, MRMIterator
from core import utils
from data import data_adapter as da
from models import model_utils
from face import core
from sklearn.neighbors import KernelDensity
import pandas as pd

import dask
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

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

def score_dataset(bandwidth, dataset, distance_threshold, conditions):
    block_size = 8000000
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

    model = model_utils.load_model('random_forest', 'adult_income')
    clf = lambda X: model.predict_proba(X)[:,1]
    conditions_function = None
    if conditions:
        X = preprocessor.transform(data)
        immutable_columns = preprocessor.get_feature_names_out(immutable_features)
        immutable_column_indices = np.arange(X.columns.shape[0])[X.columns.isin(immutable_columns)]
        conditions_function = lambda differences: core.immutable_conditions(differences, immutable_column_indices)


    print("Open a client...")
    client = None
    if not RUN_LOCALLY:
        cluster = SLURMCluster(
            processes=1,
            memory='2000MB',
            queue='defq',
            cores=2,
            walltime='00:40:00',
            log_directory=LOG_DIR
        )
        cluster.scale(16)
        client = Client(cluster)
    else:
        client = Client(n_workers=1, threads_per_worker=1)
    dask.config.set(scheduler='processes')
    dask.config.set({'temporary-directory': SCRATCH_DIR})

    data_future = client.scatter([data], broadcast=True)[0]

    face = core.Face(k_paths, clf, distance_threshold, confidence_threshold, density_threshold, conditions_function=conditions_function)

    face.set_kde(preprocessor, data, dataset, bandwidth, dir=dir)
    density_scores = face.density_scores
    density_future = client.scatter([density_scores], broadcast=True)[0]

    num_blocks = face.get_num_blocks(preprocessor, data, block_size=block_size)
    generate_graph_block = lambda block_index, data, density_scores: face.generate_graph_block(preprocessor, data, bandwidth, block_index, density_scores, dir=dir, block_size=block_size)
    futures = client.map(generate_graph_block, range(num_blocks), [data_future] * num_blocks, [density_future] * num_blocks)
    results = client.gather(futures)
    print("Finished gather results.")
    face.set_graph_from_blocks(results, preprocessor, data, dataset, bandwidth, dir=dir)
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
    conditions = bool(args[3])
    dataset = args[4]
    score_dataset(bandwidth, dataset, distance, conditions)
