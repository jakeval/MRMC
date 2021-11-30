
from data import data_adapter as da
import numpy as np
import pandas as pd


class DiceTestRunner:
    def __init__(self, N, dataset, dice, point_statistics, sort_key, k_points, immutable_features=None):
        self.sort_key = sort_key
        self.point_statistics = point_statistics
        self.dice = dice
        self.dataset = dataset
        self.N = N
        self.k_points = k_points
        if immutable_features is not None:
            self.features_to_vary = list(dataset.columns.difference(immutable_features + ['Y']))
        else:
            self.features_to_vary = list(dataset.columns.difference(['Y']))

    def run_trial(self):
        poi = da.random_poi(self.dataset)
        cf_points = self.dice.generate_counterfactuals(poi, total_CFs=self.k_points, 
            desired_class="opposite", 
            features_to_vary=self.features_to_vary).cf_examples_list[0].final_cfs_df
        cf_points = cf_points.drop('Y', axis=1)
        return self.collect_statistics(poi, cf_points), cf_points, poi

    def collect_statistics(self, poi, cf_points):
        stat_dict = {}
        for statistic, calculate_statistic in self.point_statistics.items():
            stat_dict[statistic] = calculate_statistic(poi, cf_points)
        return stat_dict

    def run_test(self):
        stats_dict_list = []
        for n in range(self.N):
            print(f"n={n}")
            stats, _, _ = self.run_trial()
            if stats is not None:
                stats_dict_list.append(stats)
        stats_dict = self.reformat_stats(stats_dict_list)
        aggregated_statistics = self.aggregate_stats(stats_dict)
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

    def aggregate_stats(self, statistics):
        statistics = self._sort_statistics(statistics)
        aggregated_statistics = {}
        for statistic in self.point_statistics:
            aggregated_statistics[statistic] = statistics[statistic].sum(axis=0) / statistics[statistic].shape[0]

        aggregated_statistics.update({'Direction ID': [k for k in range(self.k_points)]})
        return pd.DataFrame(aggregated_statistics)
    
    def _sort_statistics(self, statistics, sorted_statistic=None):
        if sorted_statistic is None:
            sorted_statistic = self.sort_key
        idx = np.argsort(-statistics[sorted_statistic], axis=1)
        sorted_statistics = statistics.copy()
        for i in range(idx.shape[0]):
            for stat in self.point_statistics:
                sorted_statistics[stat][i] = statistics[stat][i,idx[i]]
        return sorted_statistics
