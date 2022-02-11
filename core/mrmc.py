import numpy as np
from core import utils
import pandas as pd

class MRM:
    def __init__(self, alpha=utils.volcano_alpha, alpha_neg=None, ignore_negatives=True, sparsity=None,
                 perturb_dir=utils.priority_dir, weight_function=utils.centroid_normalization,
                 immutable_column_names=None, check_privacy=False):
        self.alpha = alpha
        self.alpha_neg = alpha_neg
        self.ignore_negatives = ignore_negatives
        self.weight_function = weight_function
        self.immutable_column_names = immutable_column_names
        self.X = None
        self.Y = None
        self.perturb_dir = perturb_dir
        self.check_privacy = check_privacy
        self.sparsity = sparsity

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
        original_dir = dir
        index = dir.index
        dir = pd.DataFrame(columns=index, data=[dir])
        if self.sparsity is not None:
            dir = utils.priority_dir(dir)
        if self.perturb_dir is not None:
            dir = self.perturb_dir(poi, dir)
        original_dir = pd.DataFrame(columns=original_dir.index, data=[original_dir])

        if self.immutable_column_names is not None:
            dir[self.immutable_column_names] = 0.0
            original_dir[self.immutable_column_names] = 0.0
        if self.check_privacy:
            return dir, original_dir
        return dir

    def filtered_dataset_size(self):
        return self.X[self.Y == 1].shape[0]

class MRMIterator:
    def __init__(self, mrm, preprocessor, max_iterations=100, early_stopping=None, validate=False, check_privacy=False):
        self.mrm = mrm
        self.max_iterations=max_iterations
        self.early_stopping = early_stopping
        self.preprocessor = preprocessor
        if self.early_stopping is None:
            self.early_stopping = lambda _: False
        self.validate = validate
        self.check_privacy = check_privacy

    def fit(self, dataset):
        X = dataset.drop('Y', axis=1)
        Y = dataset.Y
        X = self.preprocessor.transform(X)
        self.mrm.fit(X, Y)

    def iterate(self, poi):
        curr_poi = self.preprocessor.transform(poi).reset_index(drop=True)
        poi_path = curr_poi.copy()
        if not self.mrm.X[self.mrm.Y == 1].shape[0] > 0:
            return poi_path

        i = 0
        similarity = 0
        while i < self.max_iterations:
            dir = None
            if self.check_privacy:
                dir, original_dir = self.mrm.transform(curr_poi)
                similarity += utils.cosine_similarity(dir.to_numpy()[0], original_dir.to_numpy()[0])
            else:
                dir = self.mrm.transform(curr_poi)
            curr_poi += dir
            if np.isnan(np.array(curr_poi)).any():
                if self.check_privacy:
                    return poi_path, similarity / (poi_path.shape[0] - 1)
                return poi_path
            if self.validate:
                new_point = self.preprocessor.inverse_transform(curr_poi)
                curr_poi = self.preprocessor.transform(new_point)
            poi_path = poi_path.append(curr_poi, ignore_index=True)
            if self.early_stopping(curr_poi):
                if self.check_privacy:
                    return poi_path, similarity / (poi_path.shape[0] - 1)
                return poi_path
            i += 1
        if self.check_privacy:
            return poi_path, similarity / (poi_path.shape[0] - 1)
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
        cosine_similarities = []
        for cluster in range(self.k_dirs):
            cluster_data = self.cluster_datasets[cluster]
            super().fit(cluster_data)
            poi_path = None
            if self.check_privacy:
                poi_path, cosine_similarity = super().iterate(poi)
                cosine_similarities.append(cosine_similarity)
            else:
                poi_path = super().iterate(poi)
            paths.append(poi_path)
        if self.check_privacy:
            return paths, cosine_similarities
        return paths
        
    def get_cluster_sizes(self):
        return self.cluster_sizes
