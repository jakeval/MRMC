
from data import data_adapter as da
import numpy as np
import pandas as pd
import itertools


class DiceTestRunner:
    def __init__(self, N, dataset, preprocessor, dice, point_statistics, k_points, certainty_cutoff, immutable_features=None, feature_tolerances=None):
        self.point_statistics = point_statistics
        self.dice = dice
        self.dataset = dataset
        self.certainty_cutoff = certainty_cutoff
        self.N = N
        self.k_points = k_points
        self.feature_tolerances = feature_tolerances
        self.preprocessor = preprocessor
        if immutable_features is not None:
            self.features_to_vary = list(dataset.columns.difference(immutable_features + ['Y']))
            if feature_tolerances is not None:
                self.features_to_vary += list(feature_tolerances.keys())
        else:
            self.features_to_vary = list(dataset.columns.difference(['Y']))

    def run_trial(self):
        poi = da.random_poi(self.dataset)
        permitted_range = {}
        if self.feature_tolerances is not None:
            for key, val in self.feature_tolerances.items():
                poi_val = poi.loc[poi.index[0],key]
                permitted_range[key] = [poi_val - val, poi_val + val]
        cf_points = self.dice.generate_counterfactuals(
            poi, 
            total_CFs=self.k_points, 
            desired_class="opposite", 
            features_to_vary=self.features_to_vary,
            permitted_range=permitted_range,
            random_seed=88557,
            stopping_threshold=self.certainty_cutoff).cf_examples_list[0].final_cfs_df
        cf_points = self.preprocessor.transform(cf_points.drop('Y', axis=1))
        return self.collect_statistics(poi, cf_points), cf_points, poi

    def collect_statistics(self, poi, cf_points):
        stat_dict = {}
        for statistic, calculate_statistic in self.point_statistics.items():
            stat_dict[statistic] = calculate_statistic(self.preprocessor.transform(poi), cf_points)
        return stat_dict


    def run_test(self):
        stats_dict = dict([(stat_key, np.full(self.k_points*self.N, np.nan)) for stat_key in self.point_statistics])
        for n in range(self.N):
            i = n*self.k_points
            stats, _, _ = self.run_trial()
            for key in self.point_statistics:
                j = stats[key].shape[0] + i
                stats_dict[key][i:j] = stats[key]
        stats = pd.DataFrame(stats_dict)
        aggregated_statistics = self.aggregate_stats(stats)
        return stats, aggregated_statistics


    def reformat_stats(self, stats_dict_list):
        new_stats_dict = dict([(stat, []) for stat in stats_dict_list[0].keys()])
        # convert a list of dicts to a dict of lists
        for stat in new_stats_dict:
            for stats_dict in stats_dict_list:
                new_stats_dict[stat].append(stats_dict[stat])
        # convert a dict of lists to a dict of numpy arrays
        new_stats_dict = dict([(k, np.array(v)) for k, v in new_stats_dict.items()])
        return new_stats_dict


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
