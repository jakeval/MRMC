import numpy as np
from core import utils
import pandas as pd

class MRM:
    def __init__(self, preprocessor = None, alpha=utils.volcano_alpha,  sparsity=None,
                 perturb_dir=utils.priority_dir, weight_function=utils.centroid_normalization,
                 immutable_column_names=None, feature_tolerances=None, check_privacy=False):
        self.alpha = alpha
        self.weight_function = weight_function
        self.immutable_features = immutable_features
        self.feature_tolerances = feature_tolerances
        self.preprocessor = preprocessor
        self.immutable_column_names = preprocessor.get_feature_names_out(immutable_features)
        self.X = None
        self.Y = None
        self.perturb_dir = perturb_dir
        self.check_privacy = check_privacy
        self.sparsity = sparsity

    def filter(self, dataset, poi):
        if self.immutable_features is not None:
            filtered_data = da.filter_from_poi(dataset, poi, immutable_features=self.immutable_features, feature_tolerances=self.feature_tolerances)
        
        X = dataset.drop('Y', axis=1)
        Y = dataset.Y
        X = self.preprocessor.transform(X)
        X = X[Y == 1]
        Y = Y[Y == 1]

        if(len(X) == 0):
            raise NameError("Empty Dataset after filtering")

        if self.immutable_column_names is not None:
            X = X.drop(self.immutable_column_names, axis=1)
        self.X = X
        self.Y = Y

    def get_direction(self, poi):
        if self.immutable_column_names is not None:
            poi = poi.drop(self.immutable_column_names, axis=1)
        # indicator = (self.Y == 1) * 2 - 1
        diff = self.X - poi.to_numpy()
        dist = np.sqrt(np.power(diff, 2).sum(axis=1))
        alpha_val = self.alpha(dist)
        dir = diff.T@(alpha_val)
        if self.weight_function is not None:
            dir = self.weight_function(dir, poi.to_numpy()[0], self.X.to_numpy())
        original_dir = dir
        index = dir.index
        dir = pd.DataFrame(columns=index, data=[dir])
        # You need to fix sparsity. Must take into account categorical features when fixing sparsity
        if self.sparsity is not None:
            dir = utils.priority_dir(dir)
        if self.perturb_dir is not None:
            dir = self.perturb_dir(poi, dir)
        original_dir = pd.DataFrame(columns=original_dir.index, data=[original_dir])

        if self.immutable_column_names is not None:
            dir[self.immutable_column_names] = 0.0
            original_dir[self.immutable_column_names] = 0.0

        return dir, original_dir

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

    def fit(self, dataset, poi):
        self.mrm.filter(dataset, poi)

    def iterate(self, poi):
        curr_poi = self.preprocessor.transform(poi).reset_index(drop=True)
        poi_path = curr_poi.copy()
        # This code is redundant but can leave it here for now
        if not self.mrm.X[self.mrm.Y == 1].shape[0] > 0:
            raise NameError("Dataset size 0 after filtering")
        i = 0
        similarity = 0
        while i < self.max_iterations:
            dir = None
            dir, original_dir = self.mrm.get_direction(curr_poi)
            similarity += utils.cosine_similarity(dir.to_numpy()[0], original_dir.to_numpy()[0])
            curr_poi += dir
            if np.isnan(np.array(curr_poi)).any():
                print(poi_path)
                raise NameError("Encounted NaN in the new point")

            if self.validate:
                new_point = self.preprocessor.inverse_transform(curr_poi)
                curr_poi = self.preprocessor.transform(new_point)
            poi_path = poi_path.append(curr_poi, ignore_index=True)
            if self.early_stopping(poi_path):
                return poi_path, similarity / (poi_path.shape[0] - 1)    
            i += 1
        return poi_path, similarity / (poi_path.shape[0] - 1)

    def filtered_dataset_size(self):
        return self.mrm.filtered_dataset_size()


class MRMC:
   def __init__(self, preprocessor = None, alpha=utils.volcano_alpha, sparsity=None,
                 perturb_dir=utils.priority_dir, weight_function=utils.centroid_normalization,
                 immutable_column_names=None, feature_tolerances=None, check_privacy=False, k_dirs = 1):
        self.alpha = alpha
        self.weight_function = weight_function
        self.immutable_features = immutable_features
        self.feature_tolerances = feature_tolerances
        self.preprocessor = preprocessor
        self.immutable_column_names = preprocessor.get_feature_names_out(immutable_features)
        self.X = None
        self.Y = None
        self.perturb_dir = perturb_dir
        self.check_privacy = check_privacy
        self.sparsity = sparsity 
        self.cluster_assignments = None
        self.k_dirs = k_dirs
        self.cluster_datasets = None
        self.cluster_sizes = None

    def filter(self, dataset, cluster_assignments):
        self.cluster_assignments = cluster_assignments
        if dataset[dataset.Y == 1].shape[0] < self.k_dirs:
            raise Exception(f"The dataset is too small for the number of clusters.")
        self.cluster_datasets = []
        self.cluster_sizes = []
        for k in range(self.k_dirs):
            pos_data = dataset[dataset.Y == 1][cluster_assignments == k]
            # neg_data = dataset[dataset.Y == -1]
            # cluster_data = pd.concat([pos_data, neg_data])
            self.cluster_datasets.append(pos_data)
            self.cluster_sizes.append(pos_data.shape[0])

    def get_direction(self, poi, k):
        if k >= k_dirs:
            raise NameError("Input k values: "+str(k)+ " greater than the specified number of clusters " + str(k_dirs))

        diff = self.cluster_datasets[k] - poi.to_numpy()
        dist = np.sqrt(np.power(diff, 2).sum(axis=1))
        alpha_val = self.alpha(dist)
        dir = diff.T@(alpha_val)
        if self.weight_function is not None:
            dir = self.weight_function(dir, poi.to_numpy()[0], self.cluster_datasets[k].to_numpy())
        original_dir = dir
        index = dir.index
        dir = pd.DataFrame(columns=index, data=[dir])
        # You need to fix sparsity. Must take into account categorical features when fixing sparsity
        if self.sparsity is not None:
            dir = utils.priority_dir(dir)
        if self.perturb_dir is not None:
            dir = self.perturb_dir(poi, dir)
        original_dir = pd.DataFrame(columns=original_dir.index, data=[original_dir])

        if self.immutable_column_names is not None:
            dir[self.immutable_column_names] = 0.0
            original_dir[self.immutable_column_names] = 0.0

        return dir, original_dir





class MRMCIterator():
    def __init__(self, mrmc, preprocessor, max_iterations=100, early_stopping=None, validate=False, check_privacy=False):
        self.mrmc = mrmc
        self.max_iterations=max_iterations
        self.early_stopping = early_stopping
        self.preprocessor = preprocessor
        if self.early_stopping is None:
            self.early_stopping = lambda _: False
        self.validate = validate
        self.check_privacy = check_privacy
        

    def fit(self, dataset, cluster_assignments):
        self.mrmc.filter(dataset, cluster_assignments)

    def iterate(self, poi):
        paths = []
        cosine_similarities = []
        for cluster in range(self.k_dirs):
            curr_poi = self.preprocessor.transform(poi).reset_index(drop=True)
            poi_path = curr_poi.copy()
            if not self.mrmc.cluster_datasets[cluster].shape[0] > 0:
                raise NameError("Dataset size 0 after filtering")
            i = 0
            similarity = 0
            while i < self.max_iterations:
                dir = None
                dir, original_dir = self.mrmc.get_direction(curr_poi, cluster)
                similarity += utils.cosine_similarity(dir.to_numpy()[0], original_dir.to_numpy()[0])
                curr_poi += dir
                if np.isnan(np.array(curr_poi)).any():
                    print(poi_path)
                    raise NameError("Encounted NaN in the new point")

                if self.validate:
                    new_point = self.preprocessor.inverse_transform(curr_poi)
                    curr_poi = self.preprocessor.transform(new_point)
                poi_path = poi_path.append(curr_poi, ignore_index=True)
                if self.early_stopping(poi_path):
                    break
                i += 1
            paths.append(poi_path)
            cosine_similarities.append(similarity / (poi_path.shape[0] - 1))
        return paths
        
    def get_cluster_sizes(self):
        return self.cluster_sizes
