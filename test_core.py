import numpy as np
from data import data_adapter as da
from models import random_forest
from experiments.core import TestRunner
from core.mrmc import MRM, MRMCIterator
from experiments import path_stats
from core import utils
from visualize.two_d_plots import Display2DPaths
from matplotlib import pyplot as plt

print("Load the dataset...")
adult_train, adult_test, preprocessor = da.load_adult_income_dataset()
print("Shape before processing: ", adult_train.shape)

X = np.array(preprocessor.transform(adult_train.drop('Y', axis=1)))
Y = np.array(adult_train['Y'])

X_test = np.array(preprocessor.transform(adult_test.drop('Y', axis=1)))
Y_test = np.array(adult_test['Y'])

print("Train a model...")
model, accuracy = random_forest.train_model(X, Y, X_test, Y_test)

model_scores = model.predict_proba(X)
adult_train = da.filter_from_model(adult_train, model_scores)
print("Shape after accuracy filtering: ", adult_train.shape)

N = 5
k_dirs = 4
max_iterations = 30
immutable_features = ['age', 'sex', 'race']
feature_tolerances = {'age': 5}
immutable_column_names = preprocessor.get_feature_names_out(['age', 'sex', 'race'])
experiment_immutable_column_names = preprocessor.get_feature_names_out(['age', 'sex', 'race'])
sparsity_limit = 5
validate = True
early_stopping = lambda point: utils.model_early_stopping(model, point)
weight_function = lambda dir, poi, X: utils.centroid_normalization(dir, poi, X, alpha=0.7)
perturb_dir = None if sparsity_limit is None else lambda dir: utils.priority_dir(dir, k=sparsity_limit)
mrm = MRM(immutable_column_names=immutable_column_names, weight_function=weight_function, perturb_dir=perturb_dir)
mrmc = MRMCIterator(k_dirs, mrm, preprocessor, max_iterations, early_stopping=early_stopping, validate=validate)

positive_probability_key = 'Positive Probability'
path_statistics = {
    positive_probability_key: lambda paths: path_stats.check_positive_probability(model, paths),
    'Path Invalidity': lambda paths: path_stats.check_validity_distance(preprocessor, paths),
    'Path Count': path_stats.check_path_count,
    'Final Point Distance': path_stats.check_final_point_distance,
    'Path Length': path_stats.check_path_length,
    'Immutable Violations': lambda paths: path_stats.check_immutability(experiment_immutable_column_names, paths),
    'Sparsity': path_stats.check_sparsity,
    'Path Invalidity': lambda paths: path_stats.check_validity_distance(preprocessor, paths),
    'Diversity': path_stats.check_diversity
}
cluster_statistics = {
    'Cluster Size': path_stats.check_cluster_size,
}

test = TestRunner(N, adult_train, preprocessor, mrmc, path_statistics, 
                  cluster_statistics, positive_probability_key, 
                  immutable_features=immutable_features, 
                  feature_tolerances=feature_tolerances)
stats, aggregated_stats, nonzero_ratio = test.run_test()

print(aggregated_stats)
print(nonzero_ratio)

one_stats = True
while one_stats is None:
    one_stats, paths, clusters = test.run_trial()

    print(one_stats)

    for i, path in enumerate(paths):
        print(f"Path {i}:")
        print(preprocessor.inverse_transform(path))

    processed_paths = [np.array(path) for path in paths]

    fig, ax = Display2DPaths(X, Y).do_pca().set_paths(processed_paths).set_clusters(clusters).heatmap()
    plt.show()
    plt.close()