
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

import itertools


class MrmcTestRunner:
    def __init__(self, N, dataset, preprocessor, mrmc, path_statistics, point_statistics, cluster_statistics, pois, immutable_features=None, immutable_strict=True, feature_tolerances=None, check_privacy=False):
        self.path_statistics = path_statistics
        self.point_statistics = point_statistics
        self.cluster_statistics = cluster_statistics
        self.mrmc = mrmc
        self.statistics_keys = list(path_statistics.keys()) + list(point_statistics.keys()) + list(cluster_statistics.keys())
        self.dataset = dataset
        self.N = N
        self.preprocessor = preprocessor
        self.pois = pois
        self.k_dirs = mrmc.k_dirs
        self.immutable_features = immutable_features
        self.feature_tolerances = feature_tolerances
        self.immutable_strict = immutable_strict
        self.check_privacy = check_privacy

    def run_trial(self, poi):
        """Returns a dictionary like {stat_key: [path1_stat, path2_stat, ...], stat_key1: [...]}"""
        filtered_data = self.dataset
        if self.immutable_features is not None:
            filtered_data = da.filter_from_poi(self.dataset, poi, immutable_features=self.immutable_features, feature_tolerances=self.feature_tolerances)
        if filtered_data[filtered_data.Y == 1].shape[0] < self.k_dirs:
            if self.immutable_strict:
                return self.get_null_results(), None, None # we can't run a trial with so few data points
            else:
                filtered_data = self.dataset
        cluster_assignments, km = self.get_clusters(filtered_data, self.k_dirs)
        self.mrmc.fit(filtered_data, cluster_assignments)
        paths = None
        cosine_similarity = None
        if self.check_privacy:
            paths, cosine_similarity = self.mrmc.iterate(poi)
        else:
            paths = self.mrmc.iterate(poi)
        if paths is None:
            return self.get_null_results(), None, None
        for path in paths:
            if path is None:
                return self.get_null_results(), None, None
        stats = self.collect_statistics(poi, paths, cluster_assignments)
        if self.check_privacy:
            stats['Cosine Similarity'] = cosine_similarity
        return stats, paths, km.cluster_centers_

    def collect_statistics(self, poi, paths, cluster_assignments):
        stat_dict = {}
        for statistic, calculate_statistic in self.path_statistics.items():
            stat_dict[statistic] = calculate_statistic(paths)
        for statistic, calculate_statistic in self.point_statistics.items():
            points = pd.DataFrame(columns=paths[0].columns, data=np.zeros((len(paths), len(paths[0].columns))))
            for i in range(len(paths)):
                points.iloc[i,:] = paths[i].iloc[-1,:]
            stat_dict[statistic] = calculate_statistic(self.preprocessor.transform(poi), points)
        for statistic, calculate_statistic in self.cluster_statistics.items():
            stat_dict[statistic] = calculate_statistic(cluster_assignments, self.k_dirs)
        return stat_dict

    def get_null_results(self):
        d = {}
        for key in self.statistics_keys:
            d[key] = np.full(self.k_dirs, np.nan)
        if self.check_privacy:
            d['Cosine Similarity'] = np.full(self.k_dirs, np.nan)
        return d

    def run_test(self):
        stats_dict = dict([(stat_key, np.full(self.k_dirs*self.N, np.nan)) for stat_key in self.statistics_keys])
        if self.check_privacy:
            stats_dict['Cosine Similarity'] = np.full(self.k_dirs*self.N, np.nan)
        for n in range(self.N):
            #print(f"n={n}")
            poi = self.pois.iloc[[n],:]
            i = n*self.k_dirs
            j = (n+1)*self.k_dirs
            stats, _, _ = self.run_trial(poi)
            for key in self.statistics_keys:
                stats_dict[key][i:j] = stats[key]
            if self.check_privacy:
                stats_dict['Cosine Similarity'][i:j] = stats['Cosine Similarity']
        stats = pd.DataFrame(stats_dict)
        aggregated_statistics = self.aggregate_stats(stats)
        return stats, aggregated_statistics

    def get_clusters(self, dataset, n_clusters):
        X = np.array(self.preprocessor.transform(dataset.drop('Y', axis=1)))
        Y = np.array(dataset['Y'])
        km = KMeans(n_clusters=n_clusters)
        km.fit(X[Y == 1])
        cluster_assignments = km.predict(X[Y == 1])
        return cluster_assignments, km

    def aggregate_stats(self, statistics):
        non_null_stats = statistics[~statistics.isnull().any(axis=1)]
        non_null_ratio = non_null_stats.shape[0] / statistics.shape[0]
        meta_stats = ['mean', 'median', 'min', 'max', 'std']
        described_stats = non_null_stats.describe()
        described_stats.loc['median',:] = non_null_stats.median()
        aggregated_columns = [f'{metric} ({stat})' for metric, stat in itertools.product(statistics.columns, meta_stats)]
        aggregated_stats = pd.DataFrame(columns=aggregated_columns, data=np.zeros(shape=(1,len(aggregated_columns))))
        
        for metric, stat in itertools.product(statistics.columns, meta_stats):
            aggregated_stats[f'{metric} ({stat})'] = described_stats.loc[stat, metric]
        if self.check_privacy:
            for metric in meta_stats:
                aggregated_stats[f'Cosine Similarity ({stat})'] = described_stats.loc[metric, 'Cosine Similarity']

        aggregated_stats.loc[:,'success_ratio'] = non_null_ratio
        return aggregated_stats
