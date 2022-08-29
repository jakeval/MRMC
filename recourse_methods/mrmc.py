import numpy as np
from core import utils
import pandas as pd
from data import data_adapter as da
from sklearn.cluster import KMeans


class MRM:
    def __init__(self,
                 dataset,
                 preprocessor,
                 label_column='Y',
                 positive_label=1,
                 alpha=utils.volcano_alpha,
                 rescale_direction=None):
        self.X = MRM.process_data(dataset, preprocessor, label_column, positive_label)
        self.preprocessor = preprocessor
        self.alpha = alpha
        self.rescale_direction = rescale_direction

    @staticmethod
    def process_data(dataset, preprocessor, label_column, positive_label):
        positive_dataset = dataset[dataset[label_column] == positive_label]
        X = preprocessor.transform(positive_dataset.drop(label_column, axis=1))
        if len(X) == 0:
            raise ValueError("Dataset is empty after excluding non-positive outcome examples.")
        return X

    # TODO: how should this work for immutable columns?
    def get_unnormalized_direction(self, poi):
        columns = poi.columns
        diff = (self.X.to_numpy() - self.preprocessor.transform(poi).to_numpy())
        dist = np.sqrt(np.power(diff, 2).sum(axis=1))
        alpha_val = self.alpha(dist)
        dir = diff.T @ alpha_val
        return pd.DataFrame(columns=columns, data=dir[None,:])

    def get_recourse_direction(self, poi):
        direction = self.get_unnormalized_direction(poi)
        if self.rescale_direction:
            direction = self.rescale_direction(self, direction)
        return direction

    def get_recourse_instructions(self, poi):
        direction = self.get_recourse_direction(poi)
        return self.preprocessor.directions_to_instructions(direction)


class Clusters:
    def __init__(self,
                 cluster_assignments,
                 cluster_centers):
        self.cluster_assignments = cluster_assignments
        self.cluster_centers = cluster_centers


class MRMC:
    def __init__(self,
                 k_directions,
                 preprocessor,
                 dataset,
                 label_column='Y',
                 positive_label=1,
                 alpha=utils.volcano_alpha,
                 rescale_direction=None,
                 clusters=None):
        X = MRM.process_data(dataset, preprocessor, label_column, positive_label)
        self.k_directions = k_directions
        if not clusters:
            clusters = self.cluster_data(X, self.k_directions)
        self.clusters = clusters
        self.validate_cluster_assignments(clusters.cluster_assignments, self.k_directions)
        
        mrms = []
        for cluster_index in range(k_directions):
            X_cluster = X[clusters.cluster_assignments == cluster_index]
            dataset_cluster = dataset.loc[X_cluster.index]
            mrm = MRM(
                dataset=dataset_cluster,
                preprocessor=preprocessor,
                label_column=label_column,
                positive_label=positive_label,
                alpha=alpha,
                rescale_direction=rescale_direction)
            mrms.append(mrm)
        self.mrms = mrms

    def cluster_data(self, X, k_directions):
        km = KMeans(n_clusters=k_directions)
        cluster_assignments = km.fit_predict(X.to_numpy())
        cluster_centers = km.cluster_centers_
        return Clusters(cluster_assignments, cluster_centers)

    def validate_cluster_assignments(self, cluster_assignments, k_directions):
        cluster_assignments = pd.DataFrame({'cluster_index': cluster_assignments})
        cluster_sizes = cluster_assignments.groupby('cluster_index').count()
        if set(cluster_sizes.index) != set(range(k_directions)):
            raise RuntimeError(f'Data was assigned to clusters {cluster_sizes.index}, but expected clusters {range(k_directions)}')
        return True

    def get_all_recourse_instructions(self, poi):
        instructions = []
        for mrm in self.mrms:
            instructions.append(mrm.get_recourse_instructions(poi))
        instructions = pd.concat(instructions).reset_index(drop=True)
        return instructions

    def get_kth_recourse_instructions(self, poi, dir_index):

        return self.mrms[dir_index].get_recourse_instructions(poi)
