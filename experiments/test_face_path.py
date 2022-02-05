
from data import data_adapter as da
import numpy as np
import pandas as pd

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

    def iterate_face(self, poi, first_cf, max_iterations):
        old_poi = self.preprocessor.transform(poi)
        path = old_poi.reset_index(drop=True)
        new_cf = first_cf
        for i in range(max_iterations):
            new_cf_processed = self.preprocessor.transform(new_cf)
            dir = new_cf_processed.to_numpy() - old_poi.to_numpy()
            dir = self.weight_function(dir)
            if self.perturb_dir is not None:
                dir = self.perturb_dir(dir)
            dir[0,old_poi.columns == 'age'] = 0.0
            curr_poi = old_poi + dir
            if np.isinf(curr_poi).any().any():
                return path
            if np.isnan(curr_poi).any().any():
                return path
            curr_poi = self.preprocessor.inverse_transform(curr_poi)
            curr_poi = self.preprocessor.transform(curr_poi)
            path = path.append(curr_poi, ignore_index=True)
            model_certainty = self.clf(curr_poi.to_numpy())
            curr_poi = self.preprocessor.inverse_transform(curr_poi)
            if model_certainty >= self.face.confidence_threshold:
                return path

            new_paths = self.face.iterate_new_point(curr_poi, 1)
            if len(new_paths) == 1:
                new_cf = new_paths[0].loc[new_paths[0].index == new_paths[0].index[-1],:]
            else:
                return path
            new_cf = self.preprocessor.transform(new_cf)
        return path

    def run_trial(self, poi):
        """Returns a dictionary like {stat_key: [path1_stat, path2_stat, ...], stat_key1: [...]}"""
        poi_index = poi.index[0]
        self.face.fit(self.dataset, self.preprocessor)
        if self.immutable_features is not None:
            self.face.add_age_condition(self.age_tolerance, poi_index)
        paths = self.face.iterate(poi_index)
        perturbed_paths = []
        for path in paths:
            cf = path.loc[path.index == path.index[-1],:]
            cf2 = self.preprocessor.inverse_transform(cf)
            perturbed_paths.append(self.iterate_face(poi, cf2, self.max_iterations - 1))
        self.face.clear_age_condition()

        if perturbed_paths is None:
            return self.get_null_results(), None
        for path in perturbed_paths:
            if path is None:
                return self.get_null_results(), None
        if len(perturbed_paths) == 0:
            return self.get_null_results(), None
        if self.immutable_features is not None: # zero-out any changes to age. TODO: this should be done at each iteration
            poi_age = self.preprocessor.transform(poi).loc[poi.index[0],'age']
            for i in range(len(perturbed_paths)):
                perturbed_paths[i].loc[perturbed_paths[i].index[-1],'age'] = poi_age
        return self.collect_statistics(self.dataset.loc[[poi_index],:].drop('Y', axis=1), perturbed_paths), perturbed_paths

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
