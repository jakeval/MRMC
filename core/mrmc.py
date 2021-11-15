import numpy as np
from core import utils
import pandas as pd


#TODO: instead of passing in a preprocessor, this should create one itself using column info.
class MRM:
    def __init__(self, alpha=utils.cliff_alpha, alpha_neg=None, ignore_negatives=True, 
                 immutable_features=None, preprocessor=None, perturb_dir=utils.constant_priority_dir, 
                 weight_function=utils.size_normalization):
        self.alpha = alpha
        self.alpha_neg = alpha_neg
        self.ignore_negatives = ignore_negatives
        self.weight_function = weight_function
        if weight_function is None:
            self.weight_function = lambda dir, X: dir
        self.X = None
        self.Y = None
        self.preprocessor = preprocessor
        self.immutable_features = immutable_features
        self.perturb_dir = perturb_dir

    def fit(self, dataset):
        df = dataset.copy()
        if self.ignore_negatives:
            df = df[df.Y == 1]
        self.X = df.drop("Y", axis=1)
        self.Y = df.Y
        if self.immutable_features is not None:
            self.X = self.X.drop(self.immutable_features, axis=1)
        if self.preprocessor is not None:
            self.X = self.preprocessor.transform(self.X)

    def transform(self, poi):
        columns_to_restore = None
        if self.immutable_features is not None:
            columns_to_restore = poi[self.immutable_features]
            poi = poi.drop(self.immutable_features, axis=1)
        if self.preprocessor is not None:
            poi = self.preprocessor.transform(poi)

        indicator = (self.Y == 1) * 2 - 1
        diff = self.X - poi.to_numpy()
        dist = np.sqrt(np.power(diff, 2).sum(axis=1))
        alpha_val = self.alpha(dist)
        if not self.ignore_negatives and self.alpha_neg is not None:
            alpha_val = np.where(indicator == 1, alpha_val, self.alpha_neg(dist))
        dir = diff.T@(alpha_val * indicator)
        if self.weight_function is not None:
            dir = self.weight_function(dir, self.X)
        if self.perturb_dir is not None:
            dir = self.perturb_dir(dir)

        new_point = poi.copy()
        new_point += dir
        if self.preprocessor is not None:
            new_point = self.preprocessor.inverse_transform(new_point)
        if self.immutable_features is not None:
            new_point[self.immutable_features] = columns_to_restore
        return new_point

    def filtered_dataset_size(self):
        return self.X[self.Y == 1].shape[0]


class MRMIterator:
    def __init__(self, mrm, max_iterations=100, early_stopping=None):
        self.mrm = mrm
        self.max_iterations=max_iterations
        self.dimensions = None
        self.early_stopping = early_stopping
        if self.early_stopping is None:
            self.early_stopping = lambda _: False

    def fit(self, dataset):
        self.mrm.fit(dataset)
        self.dimensions = dataset.shape[1] - 1 # subtract one for the label dimension

    def iterate(self, poi):
        curr_poi = poi
        poi_path = pd.DataFrame(columns=poi.columns)
        i = 0
        while i < self.max_iterations:
            poi_path = poi_path.append(curr_poi, ignore_index=True)
            curr_poi = self.mrm.transform(curr_poi)
            i += 1
            if self.early_stopping(curr_poi):
                return poi_path[:i]
        return poi_path

    def filtered_dataset_size(self):
        return self.mrm.filtered_dataset_size()


class MRMCIterator(MRMIterator):
    def __init__(self, k_dirs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cluster_assignments = None
        self.k_dirs = k_dirs
        self.X = None
        self.Y = None
        self.cluster_datasets = None
        self.cluster_sizes = None

    def fit(self, dataset, cluster_assignments):
        self.cluster_assignments = cluster_assignments
        if dataset[dataset.Y == 1].shape[0] < self.k_dirs:
            raise Exception(f"The dataset is too small for the number of clusters.")
        self.cluster_datasets = []
        self.cluster_sizes = []
        for k in range(self.k_dirs):
            pos_data = dataset[dataset.Y == 1][cluster_assignments == k]
            neg_data = dataset[dataset.Y == -1]
            cluster_data = pd.concat([pos_data, neg_data])
            self.cluster_datasets.append(cluster_data)
            self.cluster_sizes.append(pos_data.shape[0])

    def iterate(self, poi):
        paths = []
        for cluster in range(self.k_dirs):
            cluster_data = self.cluster_datasets[cluster]
            super().fit(cluster_data)
            poi_path = poi.reset_index()
            if self.cluster_sizes[cluster] > 0:
                poi_path = super().iterate(poi)
            paths.append(poi_path)
        return paths
        
    def get_cluster_sizes(self):
        return self.cluster_sizes
