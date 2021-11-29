
from core.mrmc import MRM, MRMCIterator, MRMIterator
from core import utils
from data import data_adapter as da
from models import random_forest
from visualize.two_d_plots import Display2DPaths

from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

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

# Filtering for Immutability...
print("Select a random POI...")
poi = da.random_poi(adult_train, drop_label=True)
print(poi[['age', 'race', 'sex']])
do_filtering = False
immutable_features = ['age']
immutable_columns = preprocessor.get_feature_names_out(immutable_features)
feature_tolerances = {'age': 5}
if do_filtering:
    immutable_features = ['age', 'sex', 'race'] #, 'relationship', 'marital-status']
    feature_tolerances = {'age': 5}
    print(poi[immutable_features])

filtered_adult = da.filter_from_poi(adult_train, poi, immutable_features, feature_tolerances)
print("Shape after immutability filtering: ", filtered_adult.shape)
print("Positive examples after immutability filtering: ", filtered_adult[filtered_adult.Y == 1].shape)

# Set alpha values
alpha = lambda dist: utils.cliff_alpha(dist, cutoff=0.5, degree=2)

perturb_dir = None # utils.constant_step_size

print("Do clustering...")
X = np.array(preprocessor.transform(filtered_adult.drop('Y', axis=1)))
Y = np.array(filtered_adult['Y'])
n_clusters=4
km = KMeans(n_clusters=n_clusters)
km.fit(X[Y == 1])
cluster_assignments = km.predict(X[Y == 1])

run_mrm = False
if run_mrm:
    print("Run MRM...")
    # Prepare MRMC
    processed_adult = preprocessor.transform(filtered_adult)
    processed_poi = preprocessor.transform(poi)
    mrm = MRM(alpha=alpha, ignore_negatives=True,
            immutable_features=None, perturb_dir=None, 
            preprocessor=None)
    
    for k in range(n_clusters):
        cluster_center = km.cluster_centers_[k]
        diff = (cluster_center - processed_poi.to_numpy()[0])
        dist = np.sqrt(diff@diff)
        print("-"*30)
        print(f"Cluster {k}: {dist}")
        cluster_data = processed_adult[processed_adult.Y == 1][cluster_assignments == k]
        mrm.fit(cluster_data)
        newpoint = mrm.transform(processed_poi)
        diff = newpoint.to_numpy()[0] - processed_poi.to_numpy()[0]
        dist = np.sqrt(diff@diff)
        print("Step size: ", dist)
        print(newpoint)

do_validated = True
do_unvalidated = True

max_iterations = 30

if do_validated:
    weight_function = lambda dir, poi, X: utils.centroid_normalization(dir, poi, X, alpha=0.7)
    
    mrm = MRM(alpha=alpha, ignore_negatives=True, 
          immutable_column_names=immutable_columns, perturb_dir=perturb_dir,
          weight_function=weight_function)
    mrmc = MRMCIterator(n_clusters,
                        mrm,
                        preprocessor,
                        max_iterations=max_iterations,
                        validate=True)

    # Generate recourse paths
    mrmc.fit(filtered_adult, cluster_assignments)
    print("Generating paths...")
    paths = mrmc.iterate(poi)

    X_full = np.array(preprocessor.transform(adult_train.drop('Y', axis=1)))
    Y_full = np.array(adult_train['Y'])

    paths_processed = []
    for i, path in enumerate(paths):
        paths_processed.append(path.to_numpy())
        final_point = preprocessor.inverse_transform(path.iloc[-1:])
        print("-"*30)
        print(f"Path {i}")
        for column in poi.columns:
            print(f"{column}: {poi[column].iloc[0]} -> {final_point[column]}")

    display = Display2DPaths(X_full, Y_full, title="UCI Adult Income Demo")
    fig, ax = display.do_pca().set_clusters(km.cluster_centers_).set_paths(paths_processed).heatmap()
    plt.show()
    plt.close()

if do_unvalidated:

    processed_poi = preprocessor.transform(poi)

    weight_function = lambda dir, poi, X: utils.centroid_normalization(dir, poi, X, alpha=0.7)

    print("Run preprocessed MRMC")
    mrm = MRM(alpha=alpha, ignore_negatives=True, 
            immutable_column_names=immutable_columns, perturb_dir=perturb_dir, 
            weight_function=weight_function)
    
    mrmc = MRMCIterator(n_clusters,
                        mrm,
                        preprocessor,
                        max_iterations=max_iterations,
                        validate=False)

    # Generate recourse paths
    mrmc.fit(filtered_adult, cluster_assignments)
    print("Generating paths...")
    paths = mrmc.iterate(poi)
    processed_paths = []
    print_validated = True
    print_unvalidated = True
    if print_validated:
        for i, path in enumerate(paths):
            processed_paths.append(path.to_numpy())

            final_point = path.iloc[-1:]
            print("-"*30)
            print(f"Path {i}")
            final_point = preprocessor.inverse_transform(final_point)
            for column in poi.columns:
                print(f"{column}: {poi[column].iloc[0]} -> {final_point[column].iloc[0]}")
    if print_unvalidated:
        for i, path in enumerate(paths):
            processed_paths.append(path.to_numpy())
            final_point = path.iloc[-1:]
            print("-"*30)
            print(f"Path {i}")
            for column in processed_poi.columns:
                print(f"{column}: {processed_poi[column].iloc[0]} -> {final_point[column].iloc[0]}")

    X_full = np.array(preprocessor.transform(adult_train.drop('Y', axis=1)))
    Y_full = np.array(adult_train['Y'])

    display = Display2DPaths(X_full, Y_full, title="UCI Adult Income Demo")
    fig, ax = display.do_pca().set_clusters(km.cluster_centers_).set_paths(processed_paths).heatmap()
    plt.show()
    plt.close()
