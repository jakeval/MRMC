import numpy as np
from core import utils

"""
Options:

1. Have it return a new point, not a direction
    > This means it will need a direction perturbation parameter
2. Have it return a raw direction
3. Have it convert the direction
    > Input points will always have 0/1 encoding
    > If we generate a direction saying +- 0, return what?
    > Actually, returning a raw direction makes sense:
        - tells you how to change continuous variables
        - tells you how to change binary variables (change from 100% male to 90% male, 10% female)
        - tells you how to change categorical variables
            (change from 100% private sector to 0% private sector, 50% local gov, 50% federal gov)
"""
#TODO: instead of passing in a preprocessor, this should create one itself using column info.
class MRM:
    def __init__(self, alpha=utils.cliff_alpha, alpha_neg=None, ignore_negatives=True, immutable_features=None, preprocessor=None, perturb_dir=utils.constant_priority_dir, weight_function=utils.size_normalization):
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
        for column in new_point.columns:
            print(f"{column}: {poi[column].iloc[0]} -> {new_point[column].iloc[0]}")
        if self.preprocessor is not None:
            new_point = self.preprocessor.inverse_transform(new_point)
        if self.immutable_features is not None:
            new_point[self.immutable_features] = columns_to_restore
        return new_point

    def filtered_dataset_size(self):
        return self.X[self.Y == 1].shape[0]


class MRMIterator:
  def __init__(self, mrm, max_iterations=100, perturb_dir=utils.constant_priority_dir, early_stopping=None):
    self.mrm = mrm
    self.perturb_dir = perturb_dir
    if perturb_dir is None:
      self.perturb_dir = lambda dir: dir
    self.max_iterations=max_iterations
    self.dimensions = None
    self.early_stopping = early_stopping
    if self.early_stopping is None:
      self.early_stopping = lambda _: False

  def fit(self, X, Y):
    self.mrm.fit(X, Y)
    self.dimensions = X.shape[1]

  def iterate(self, poi):
    curr_poi = poi
    poi_path = np.empty(shape=(self.max_iterations, self.dimensions))
    i = 0
    while i < self.max_iterations:
      poi_path[i] = curr_poi
      dir = self.mrm.transform(curr_poi)
      dir = self.perturb_dir(dir)
      curr_poi = curr_poi + dir
      i += 1
      if self.early_stopping(curr_poi):
        return poi_path[:i]
    return poi_path[:i]

  def filtered_dataset_size(self):
    return self.mrm.filtered_dataset_size()


class MRMCIterator(MRMIterator):
  def __init__(self, clusterer, k_dirs, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.cluster_centers_ = None
    self.cluster_assignments_ = None
    self.clusterer = clusterer
    self.k_dirs = k_dirs
    self.X = None
    self.Y = None
    self.cluster_datasets = None
    self.filtered_dataset_sizes = None

  def fit(self, X, Y, recalculate_clusters=False):
    if recalculate_clusters:
      self.clusterer.fit(X[Y == 1])
    self.cluster_assignments_ = self.clusterer.predict(X[Y == 1])
    self.cluster_centers_ = self.clusterer.cluster_centers_
    self.X = X
    self.Y = Y

    self.cluster_datasets = []
    self.filtered_dataset_sizes = np.zeros(self.k_dirs)
    for i in range(self.k_dirs):
      X_cluster = X[Y == 1][self.cluster_assignments_ == i]
      Y_cluster = Y[Y == 1][self.cluster_assignments_ == i]
      X_full = np.concatenate([X[Y == -1], X_cluster])
      Y_full = np.concatenate([Y[Y == -1], Y_cluster])
      self.cluster_datasets.append((X_full, Y_full))
      super().fit(X_full, Y_full)
      self.filtered_dataset_sizes[i] = super().filtered_dataset_size()
    
  def iterate(self, poi):
    paths = []
    for X, Y in self.cluster_datasets:
      super().fit(X, Y)
      poi_path = poi[None,:]
      if super().filtered_dataset_size() > 0:
        poi_path = super().iterate(poi)
      paths.append(poi_path)
    return paths

  def filtered_dataset_size(self):
    return self.filtered_dataset_sizes