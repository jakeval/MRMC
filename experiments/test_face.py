
from data import data_adapter as da
import numpy as np
import pandas as pd

import itertools


class FaceTestRunner:
    def __init__(self, N, dataset, preprocessor, face, point_statistics, k_points, immutable_features=None, age_tolerance=None):
        self.point_statistics = point_statistics
        self.face = face
        self.dataset = dataset
        self.N = N
        self.preprocessor = preprocessor
        self.k_points = k_points
        self.immutable_features = immutable_features
        self.age_tolerance = age_tolerance

    def run_trial(self):
        """Returns a dictionary like {stat_key: [path1_stat, path2_stat, ...], stat_key1: [...]}"""
        poi = da.random_poi(self.dataset)
        poi_index = poi.index[0]
        self.face.fit(self.dataset, self.preprocessor)
        if self.age_tolerance is not None:
            self.face.add_age_condition(self.age_tolerance, poi_index)
        paths = self.face.iterate(poi_index)
        self.face.clear_age_condition()
        if paths is None:
            return self.get_null_results(), None
        for path in paths:
            if path is None:
                return self.get_null_results(), None
        if len(paths) == 0:
            return self.get_null_results(), None
        if self.immutable_features is not None: # zero-out any changes to age.
            poi_age = self.preprocessor.transform(poi).loc[poi.index[0],'age']
            for i in range(len(paths)):
                paths[i].loc[paths[i].index[-1],'age'] = poi_age
        return self.collect_statistics(self.dataset.loc[[poi_index],:].drop('Y', axis=1), paths), paths

    def collect_statistics(self, poi, paths):
        stat_dict = {}
        for statistic, calculate_statistic in self.point_statistics.items():
            points = pd.DataFrame(columns=paths[0].columns, data=np.zeros((len(paths), len(paths[0].columns))))
            for i in range(len(paths)):
                points.iloc[i,:] = paths[i].iloc[-1,:]
            stat_dict[statistic] = calculate_statistic(self.preprocessor.transform(poi), points)
        return stat_dict

    def get_null_results(self):
        d = {}
        for key in self.point_statistics:
            d[key] = np.full(self.k_points, np.nan)
        return d

    def run_test(self):
        stats_dict = dict([(stat_key, np.full(self.k_points*self.N, np.nan)) for stat_key in self.point_statistics])
        for n in range(self.N):
            i = n*self.k_points
            stats, _ = self.run_trial()
            for key in self.point_statistics:
                j = stats[key].shape[0] + i
                stats_dict[key][i:j] = stats[key]
        stats = pd.DataFrame(stats_dict)
        aggregated_statistics = self.aggregate_stats(stats)
        return stats, aggregated_statistics

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

        aggregated_stats.loc[:,'success_ratio'] = non_null_ratio
        return aggregated_stats
