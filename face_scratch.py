
from core.mrmc import MRM, MRMCIterator, MRMIterator
from core import utils
from data import data_adapter as da
from models import model_utils
from visualize.two_d_plots import Display2DPaths
from face import core
from sklearn.neighbors import KernelDensity
import pandas as pd

from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import sys

RUN_LOCALLY = True
dir = '/home/jasonvallada/MRMC/face_density'
if RUN_LOCALLY:
    dir = './face_density'

def score_dataset(bandwidth, dataset):
    print(f"Run with bandwidth {bandwidth} and dataset {dataset}")
    print("Load the dataset...")
    data = None
    preprocessor = None
    if dataset == 'adult_income':
        data, _, preprocessor = da.load_adult_income_dataset()
    else:
        data, _, preprocessor = da.load_german_credit_dataset()

    k_paths = 4
    distance_threshold=1.2
    confidence_threshold = 0.7
    density_threshold = 1

    clf = None

    face = core.Face(k_paths, clf, distance_threshold, confidence_threshold, density_threshold)
    face.set_kde(preprocessor, data, dataset, bandwidth, dir=dir)
    scores = face.density_scores
    scores_df = pd.DataFrame({'scores': scores})
    print(scores_df.describe())


if __name__ == '__main__':
    args = sys.argv
    bandwidth = float(args[1])
    dataset = args[2]
    score_dataset(bandwidth, dataset)
