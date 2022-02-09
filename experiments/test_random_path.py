from data import data_adapter as da
import numpy as np
import pandas as pd
import itertools
from experiments import recourse_iterator


class RandomPathTestRunner:
    def __init__(self, N, dataset, preprocessor, std_dev,
                 point_statistics, path_statistics, k_points,
                 certainty_cutoff, max_iterations,
                 weight_function, clf, pois, perturb_dir=None, immutable_features=None):
        self.point_statistics = point_statistics
        self.path_statistics = path_statistics
        self.statistics_keys = list(path_statistics.keys()) + list(point_statistics.keys())
        self.std_dev = std_dev
        self.dataset = dataset
        self.certainty_cutoff = certainty_cutoff
        self.N = N
        self.k_points = k_points
        self.perturb_dir = perturb_dir
        self.max_iterations = max_iterations
        self.preprocessor = preprocessor
        self.weight_function = weight_function
        self.clf = clf
        self.pois = pois
        if immutable_features is not None:
            self.features_to_vary = list(dataset.columns.difference(immutable_features + ['Y']))
        else:
            self.features_to_vary = list(dataset.columns.difference(['Y']))

    def get_random_recourse(self, transformed_poi, k_points):
        transformed_columns = self.preprocessor.get_feature_names_out(self.features_to_vary)
        r = np.random.normal(0,self.std_dev, (k_points,len(transformed_columns)))
        cf_points = pd.concat([transformed_poi]*k_points)
        cf_points[transformed_columns] += r
        return cf_points

    def run_trial(self, poi):
        transformed_poi = self.preprocessor.transform(poi)
        paths = recourse_iterator.iterate_recourse(
            transformed_poi,
            self.preprocessor,
            self.max_iterations - 1,
            self.certainty_cutoff,
            self.k_points,
            self.get_random_recourse,
            self.clf,
            self.weight_function,
            perturb_dir=self.perturb_dir
        )

        return self.collect_statistics(poi, paths), paths, poi

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

    def run_test(self):
        stats_dict = dict([(stat_key, np.full(self.k_points*self.N, np.nan)) for stat_key in self.statistics_keys])
        for n, poi_index in zip(range(self.N), self.pois):
            i = n*self.k_points
            j = (n+1)*self.k_points
            poi = self.dataset[self.dataset.index == poi_index].drop('Y', axis=1)
            stats, _, _ = self.run_trial(poi)
            for key in self.statistics_keys:
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
