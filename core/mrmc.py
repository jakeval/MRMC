import numpy as np
from core import utils
import pandas as pd


#TODO: instead of passing in a preprocessor, this should create one itself using column info.
class MRM:
    def __init__(self, alpha=utils.cliff_alpha, alpha_neg=None, ignore_negatives=True, 
                 perturb_dir=utils.priority_dir, weight_function=utils.centroid_normalization,
                 immutable_column_names=None):
        self.alpha = alpha
        self.alpha_neg = alpha_neg
        self.ignore_negatives = ignore_negatives
        self.weight_function = weight_function
        self.immutable_column_names = immutable_column_names
        self.X = None
        self.Y = None
        self.perturb_dir = perturb_dir

    def fit(self, X, Y):
        if self.ignore_negatives:
            X = X[Y == 1]
            Y = Y[Y == 1]
        if self.immutable_column_names is not None:
            X = X.drop(self.immutable_column_names, axis=1)
        self.X = X
        self.Y = Y

    def transform(self, poi):
        if self.immutable_column_names is not None:
            poi = poi.drop(self.immutable_column_names, axis=1)
        indicator = (self.Y == 1) * 2 - 1
        diff = self.X - poi.to_numpy()
        dist = np.sqrt(np.power(diff, 2).sum(axis=1))
        alpha_val = self.alpha(dist)
        if not self.ignore_negatives and self.alpha_neg is not None:
            alpha_val = np.where(indicator == 1, alpha_val, self.alpha_neg(dist))
        dir = diff.T@(alpha_val * indicator)
        if self.weight_function is not None:
            dir = self.weight_function(dir, poi.to_numpy()[0], self.X.to_numpy())
        if self.perturb_dir is not None:
            dir = self.perturb_dir(dir)
        
        dir = pd.DataFrame(columns=dir.index, data=[dir])

        if self.immutable_column_names is not None:
            dir[self.immutable_column_names] = 0.0
        return dir

    def filtered_dataset_size(self):
        return self.X[self.Y == 1].shape[0]


class MRMIterator:
    def __init__(self, mrm, preprocessor, max_iterations=100, early_stopping=None, validate=True):
        self.mrm = mrm
        self.max_iterations=max_iterations
        # self.dimensions = None
        self.early_stopping = early_stopping
        self.preprocessor = preprocessor
        if self.early_stopping is None:
            self.early_stopping = lambda _: False
        self.validate = validate

    def fit(self, dataset):
        X = dataset.drop('Y', axis=1)
        Y = dataset.Y
        X = self.preprocessor.transform(X)
        self.mrm.fit(X, Y)
        # self.dimensions = dataset.shape[1] - 1 # subtract one for the label dimension

    def iterate(self, poi):
        # print("-"*10, "Start Iteration", "-"*10)
        curr_poi = self.preprocessor.transform(poi).reset_index(drop=True)
        path_columns = curr_poi.columns
        if self.validate:
            path_columns = poi.columns
        poi_path = pd.DataFrame(columns=path_columns, dtype='float64')
        i = 0

        occ_columns = self.preprocessor.get_feature_names_out(['occupation'])

        while i < self.max_iterations:
            new_point = curr_poi
            original = None
            if self.validate:
                new_point = self.preprocessor.inverse_transform(new_point)
                # print(new_point['occupation'].values)
                curr_poi = self.preprocessor.transform(new_point)
            original = curr_poi[occ_columns].idxmax(axis=1)
            poi_path = poi_path.append(new_point, ignore_index=True)
            i += 1
            if self.early_stopping(curr_poi):
                return poi_path[:i]
            dir = self.mrm.transform(curr_poi)
            #print(dir[occ_columns])
            #print("max:", dir[occ_columns].max())
            #print("original: ", dir[original])
            curr_poi += dir
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
            poi_path = poi.reset_index(drop=True)
            if self.cluster_sizes[cluster] > 0:
                poi_path = super().iterate(poi)
            paths.append(poi_path)
        return paths
        
    def get_cluster_sizes(self):
        return self.cluster_sizes
