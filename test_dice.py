import numpy as np
from data import data_adapter as da
from models import random_forest
from experiments.test_dice import DiceTestRunner
from experiments import point_stats
from sklearn.pipeline import Pipeline
import dice_ml
from visualize.two_d_plots import Display2DPaths
from matplotlib import pyplot as plt


"""
File for testing DiCE on a single parameter setting. To be replaced with a pattern resembling alpha_test.py
"""

print("Load the dataset...")
adult_train, adult_test, preprocessor = da.load_adult_income_dataset()
print("Shape before processing: ", adult_train.shape)

X = np.array(preprocessor.transform(adult_train.drop('Y', axis=1)))
Y = np.array(adult_train['Y'])

X_test = np.array(preprocessor.transform(adult_test.drop('Y', axis=1)))
Y_test = np.array(adult_test['Y'])

print("Train a model...")
model, accuracy = random_forest.train_model(X, Y, X_test, Y_test)
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

d = dice_ml.Data(dataframe=adult_train, continuous_features=preprocessor.continuous_features, outcome_name='Y')
m = dice_ml.Model(model=clf, backend='sklearn')

N = 40
k_points = 4

dice_random = dice_ml.Dice(d, m, method='random')

immutable_features = ['age', 'sex', 'race']
experiment_immutable_features = preprocessor.get_feature_names_out(['age', 'sex', 'race'])

positive_probability_key = 'Positive Probability'
point_statistics = {
    positive_probability_key: lambda poi, cf_points: point_stats.check_positive_probability(clf, poi, cf_points),
    'Final Point Distance': lambda poi, cf_points: point_stats.check_final_point_distance(preprocessor, poi, cf_points),
    'Immutable Violations': lambda poi, cf_points: point_stats.check_immutability(preprocessor, experiment_immutable_features, poi, cf_points),
    'Sparsity': lambda poi, cf_points: point_stats.check_sparsity(preprocessor, poi, cf_points),
    'Path Invalidity': point_stats.check_validity_distance,
    'Diversity': lambda poi, cf_points: point_stats.check_diversity(preprocessor, poi, cf_points)
}

test = DiceTestRunner(N, adult_train, dice_random, point_statistics, 
                positive_probability_key, k_points,
                immutable_features=immutable_features)
stats, aggregated_stats, nonzero_ratio = test.run_test()

print(aggregated_stats)
print(nonzero_ratio)

one_stats = True
while one_stats is None:
    one_stats, cf_points, poi = test.run_trial()

    print(one_stats)
    print(poi)
    print(cf_points)

    processed_poi = preprocessor.transform(poi).to_numpy()
    processed_points = preprocessor.transform(cf_points).to_numpy()
    paths = []
    for i in range(processed_points.shape[0]):
        path = np.array([processed_poi[0], processed_points[i]])
        paths.append(path)
    fig, ax = Display2DPaths(X, Y).do_pca().set_paths(paths).heatmap()
    plt.show()
    plt.close()

