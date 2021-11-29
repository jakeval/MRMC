
from numpy.core.fromnumeric import nonzero
from sklearn import cluster
from data import data_adapter as da
from experiments import path_stats

from core.mrmc import MRM, MRMCIterator
from core import utils

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class TestRunner:
    def __init__(self, N, dataset, preprocessor, mrmc, path_statistics, cluster_statistics, sort_key, immutable_features=None, feature_tolerances=None):
        self.sort_key = sort_key
        self.path_statistics = path_statistics
        self.cluster_statistics = cluster_statistics
        self.mrmc = mrmc
        self.statistics_keys = list(path_statistics.keys()) + list(cluster_statistics.keys())
        self.dataset = dataset
        self.N = N
        self.preprocessor = preprocessor
        self.k_dirs = mrmc.k_dirs
        self.immutable_features = immutable_features
        self.feature_tolerances = feature_tolerances

    def run_trial(self):
        poi = da.random_poi(self.dataset)
        filtered_data = da.filter_from_poi(self.dataset, poi, immutable_features=self.immutable_features, feature_tolerances=self.feature_tolerances)
        if filtered_data[filtered_data.Y == 1].shape[0] < self.k_dirs:
            return None, None, None # we can't run a trial with so few data points
        cluster_assignments, km = self.get_clusters(filtered_data, self.k_dirs)
        self.mrmc.fit(filtered_data, cluster_assignments)
        paths = self.mrmc.iterate(poi)
        return self.collect_statistics(paths, cluster_assignments), paths, km.cluster_centers_

    def collect_statistics(self, paths, cluster_assignments):
        stat_dict = {}
        for statistic, calculate_statistic in self.path_statistics.items():
            stat_dict[statistic] = calculate_statistic(paths)
        for statistic, calculate_statistic in self.cluster_statistics.items():
            stat_dict[statistic] = calculate_statistic(cluster_assignments, self.k_dirs)
        return stat_dict

    def run_test(self):
        stats_dict_list = []
        for n in range(self.N):
            print(f"n={n}")
            stats, _, _ = self.run_trial()
            if stats is not None:
                stats_dict_list.append(stats)
        stats_dict = self.reformat_stats(stats_dict_list)
        aggregated_statistics = self.aggregate_stats(stats_dict, self.mrmc.k_dirs)
        nonzero_ratio = len(stats_dict_list) / self.N

        return stats_dict, aggregated_statistics, nonzero_ratio

    def reformat_stats(self, stats_dict_list):
        new_stats_dict = dict([(stat, []) for stat in stats_dict_list[0].keys()])
        # convert a list of dicts to a dict of lists
        for stat in new_stats_dict:
            for stats_dict in stats_dict_list:
                new_stats_dict[stat].append(stats_dict[stat])
        # convert a dict of lists to a dict of numpy arrays
        new_stats_dict = dict([(k, np.array(v)) for k, v in new_stats_dict.items()])
        return new_stats_dict

    def get_clusters(self, dataset, n_clusters):
        X = np.array(self.preprocessor.transform(dataset.drop('Y', axis=1)))
        Y = np.array(dataset['Y'])
        km = KMeans(n_clusters=n_clusters)
        km.fit(X[Y == 1])
        cluster_assignments = km.predict(X[Y == 1])
        return cluster_assignments, km

    def aggregate_stats(self, statistics, k_dirs):
        statistics = self._sort_statistics(statistics)
        aggregated_statistics = {}
        for statistic in self.statistics_keys:
            aggregated_statistics[statistic] = statistics[statistic].sum(axis=0) / statistics[statistic].shape[0]

        aggregated_statistics.update({'Direction ID': [k for k in range(k_dirs)]})
        return pd.DataFrame(aggregated_statistics)
    
    def _sort_statistics(self, statistics, sorted_statistic=None):
        if sorted_statistic is None:
            sorted_statistic = self.sort_key
        idx = np.argsort(-statistics[sorted_statistic], axis=1)
        sorted_statistics = statistics.copy()
        for i in range(idx.shape[0]):
            for stat in self.statistics_keys:
                sorted_statistics[stat][i] = statistics[stat][i,idx[i]]
        return sorted_statistics
