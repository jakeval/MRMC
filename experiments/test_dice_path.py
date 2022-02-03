from data import data_adapter as da
import numpy as np
import pandas as pd
import itertools


class DicePathTestRunner:
    def __init__(self, N, dataset, preprocessor, dice,
                 point_statistics, path_statistics, k_points, 
                 certainty_cutoff, max_iterations, 
                 weight_function, clf, pois, perturb_dir=None, immutable_features=None, 
                 feature_tolerances=None):
        self.point_statistics = point_statistics
        self.path_statistics = path_statistics
        self.statistics_keys = list(path_statistics.keys()) + list(point_statistics.keys())
        self.dice = dice
        self.dataset = dataset
        self.certainty_cutoff = certainty_cutoff
        self.N = N
        self.k_points = k_points
        self.feature_tolerances = feature_tolerances
        self.perturb_dir = perturb_dir
        self.max_iterations = max_iterations
        self.preprocessor = preprocessor
        self.weight_function = weight_function
        self.clf = clf
        self.pois = pois
        if immutable_features is not None:
            self.features_to_vary = list(dataset.columns.difference(immutable_features + ['Y']))
            if feature_tolerances is not None:
                self.features_to_vary += list(feature_tolerances.keys())
        else:
            self.features_to_vary = list(dataset.columns.difference(['Y']))

    def iterate_dice(self, poi, first_cf, permitted_range, max_iterations):
        old_poi = self.preprocessor.transform(poi)
        path = old_poi.reset_index(drop=True)
        new_cf = first_cf
        for i in range(max_iterations):
            new_cf_processed = self.preprocessor.transform(new_cf)
            dir = new_cf_processed.to_numpy() - old_poi.to_numpy()
            dir = self.weight_function(dir)
            if self.perturb_dir is not None:
                dir = self.perturb_dir(dir)
            curr_poi = old_poi + dir
            if np.isinf(curr_poi).any().any():
                return path
            if np.isnan(curr_poi).any().any():
                return path
            path = path.append(curr_poi, ignore_index=True)
            curr_poi = self.preprocessor.inverse_transform(curr_poi)
            model_certainty = self.clf.predict_proba(curr_poi)[0,1]
            if model_certainty >= self.certainty_cutoff:
                return path

            desired_class = "opposite"
            if self.clf.predict_proba(curr_poi)[0,1] >= 0.5:
                desired_class = 1
            new_cf = self.dice.generate_counterfactuals(
                curr_poi, 
                total_CFs=1,
                desired_class=desired_class,
                features_to_vary=self.features_to_vary,
                permitted_range=permitted_range,
                random_seed=88557,
                stopping_threshold=self.certainty_cutoff).cf_examples_list[0].final_cfs_df
            new_cf = self.preprocessor.transform(new_cf.drop('Y', axis=1))
        return path

    def run_trial(self, poi):
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
        
        paths = []
        for i in cf_points.index:
            cf = cf_points.loc[cf_points.index == i,:].drop('Y', axis=1)
            path = self.iterate_dice(poi, cf, permitted_range, self.max_iterations - 1)
            paths.append(path)
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
