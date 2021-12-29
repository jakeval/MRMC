
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


class MrmcTestRunner:
    def __init__(self, N, dataset, preprocessor, mrmc, path_statistics, cluster_statistics, immutable_features=None, feature_tolerances=None):
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
        """Returns a dictionary like {stat_key: [path1_stat, path2_stat, ...], stat_key1: [...]}"""
        poi = da.random_poi(self.dataset)
        filtered_data = self.dataset
        if self.immutable_features is not None:
            filtered_data = da.filter_from_poi(self.dataset, poi, immutable_features=self.immutable_features, feature_tolerances=self.feature_tolerances)
        if filtered_data[filtered_data.Y == 1].shape[0] < self.k_dirs:
            return np.full(self.k_dirs, np.nan), None, None # we can't run a trial with so few data points
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
        # stats_dict_list = []
        stats_dict = dict([(stat_key, np.empty(self.k_dirs*self.N)) for stat_key in self.statistics_keys])
        for n in range(self.N):
            print(f"n={n}")
            i = n*self.k_dirs
            j = (n+1)*self.k_dirs
            print(i, j)
            stats, _, _ = self.run_trial()
            for key in self.statistics_keys:
                stats_dict[key][i:j] = stats[key]
            #if stats is not None:
            #    stats_dict_list.append(stats)
        #stats_dict = self.reformat_stats(stats_dict_list)
        stats = pd.DataFrame(stats_dict)
        aggregated_statistics = self.aggregate_stats(stats)
        nonzero_ratio = aggregated_statistics.loc['count',:][0] / (self.N*self.k_dirs)

        return stats_dict, aggregated_statistics, nonzero_ratio

    def get_clusters(self, dataset, n_clusters):
        X = np.array(self.preprocessor.transform(dataset.drop('Y', axis=1)))
        Y = np.array(dataset['Y'])
        km = KMeans(n_clusters=n_clusters)
        km.fit(X[Y == 1])
        cluster_assignments = km.predict(X[Y == 1])
        return cluster_assignments, km

    def aggregate_stats(self, statistics):
        non_null_stats = statistics[~statistics.isnull()]
        aggregated_stats = non_null_stats.describe()
        aggregated_stats.loc['median', :] = non_null_stats.median()
        return aggregated_stats
