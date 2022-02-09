
from data import data_adapter as da
import numpy as np
import pandas as pd
from experiments import recourse_iterator

import itertools


class FacePathTestRunner:
    def __init__(self,
                 N,
                 dataset,
                 preprocessor,
                 face,
                 point_statistics,
                 path_statistics,
                 k_points,
                 weight_function,
                 clf,
                 perturb_dir,
                 max_iterations,
                 pois,
                 immutable_features=None,
                 age_tolerance=None):
        self.point_statistics = point_statistics
        self.path_statistics = path_statistics
        self.statistics_keys = list(path_statistics.keys()) + list(point_statistics.keys())
        self.face = face
        self.dataset = dataset
        self.N = N
        self.preprocessor = preprocessor
        self.k_points = k_points
        self.immutable_features = immutable_features
        self.age_tolerance = age_tolerance
        self.weight_function = weight_function
        self.clf = clf
        self.perturb_dir = perturb_dir
        self.max_iterations = max_iterations
        self.pois = pois

    def get_recourse(self, poi, k_paths):
        points = pd.DataFrame(columns=poi.columns)
        face_paths = self.face.iterate_new_point(self.preprocessor.inverse_transform(poi), k_paths)
        if face_paths is None:
            return None
        if len(face_paths) == 0:
            return None
        for path in face_paths:
            points = points.append(path.iloc[[-1]], ignore_index=True)
        return points

    def run_trial(self, poi):
        """Returns a dictionary like {stat_key: [path1_stat, path2_stat, ...], stat_key1: [...]}"""

        transformed_poi = self.preprocessor.transform(poi)
        poi_index = poi.index[0]

        self.face.fit(self.dataset, self.preprocessor)
        if self.immutable_features is not None:
            ageless_immutable_features = list(filter(lambda feature: feature != 'age', self.immutable_features))
            self.face.add_age_condition(self.age_tolerance, poi_index, other_features=ageless_immutable_features)

        get_recourse = lambda poi, k_paths: self.preprocessor.transform(
            self.face.iterate_new_point(poi, k_paths).iloc[[-1]])
        get_recourse = self.get_recourse

        paths = recourse_iterator.iterate_recourse(
            transformed_poi,
            self.preprocessor,
            self.max_iterations - 1,
            self.face.confidence_threshold,
            self.k_points,
            get_recourse,
            self.clf,
            self.weight_function,
            perturb_dir=self.perturb_dir
        )

        self.face.clear_age_condition()

        if paths is None:
            return self.get_null_results(), None
        for path in paths:
            if path is None:
                return self.get_null_results(), None
        if len(paths) == 0:
            return self.get_null_results(), None

        return self.collect_statistics(self.dataset.loc[[poi_index],:].drop('Y', axis=1), paths), paths

    def collect_statistics(self, poi, paths):
        stat_dict = {}
        for statistic, calculate_statistic in self.path_statistics.items():
            stat_dict[statistic] = calculate_statistic(paths)
        for statistic, calculate_statistic in self.point_statistics.items():
            points = pd.DataFrame(columns=paths[0].columns, data=np.zeros((len(paths), len(paths[0].columns))))
            for i in range(len(paths)):
                points.iloc[i,:] = paths[i].iloc[-1,:]
            stat_dict[statistic] = calculate_statistic(self.preprocessor.transform(poi), points)
        return stat_dict

    def get_null_results(self):
        d = {}
        for key in self.statistics_keys:
            d[key] = np.full(self.k_points, np.nan)
        return d

    def run_test(self):
        stats_dict = dict([(stat_key, np.full(self.k_points*self.N, np.nan)) for stat_key in self.statistics_keys])
        for n, poi_index in zip(range(self.N), self.pois):
            i = n*self.k_points
            poi = self.dataset[self.dataset.index == poi_index].drop('Y', axis=1)
            stats, _ = self.run_trial(poi)
            for key in self.statistics_keys:
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
