
from core.mrmc import MRM, MRMCIterator
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
do_filtering = False
immutable_features = None
feature_tolerances = None
if do_filtering:
    immutable_features = ['age', 'sex', 'race'] #, 'relationship', 'marital-status']
    feature_tolerances = {'age': 5}
    print(poi[immutable_features])

filtered_adult = da.filter_from_poi(adult_train, poi, immutable_features, feature_tolerances)
print("Shape after immutability filtering: ", filtered_adult.shape)
print("Positive examples after immutability filtering: ", filtered_adult[filtered_adult.Y == 1].shape)

# Set alpha values
alpha = lambda dist: utils.cliff_alpha(dist, cutoff=0.5, degree=2)

perturb_dir = utils.constant_step_size

print("Run MRM...")
# Prepare MRMC
mrm = MRM(alpha=alpha, ignore_negatives=True, 
          immutable_features=immutable_features, perturb_dir=perturb_dir, 
          preprocessor=preprocessor)

run_mrm = False
if run_mrm:
    mrm.fit(filtered_adult)
    newpoint = mrm.transform(poi)
    print(newpoint)

    for column in poi.columns:
        print(f"{column}: {poi[column].iloc[0]} -> {newpoint[column].iloc[0]}")

print("Do clustering...")
X = np.array(preprocessor.transform(filtered_adult.drop('Y', axis=1)))
Y = np.array(filtered_adult['Y'])
n_clusters=3
km = KMeans(n_clusters=n_clusters)
km.fit(X[Y == 1])
cluster_assignments = km.predict(X[Y == 1])

mrmc = MRMCIterator(n_clusters,
                    mrm,
                    max_iterations=40)

# Generate recourse paths
mrmc.fit(filtered_adult, cluster_assignments)
print("Generating paths...")
paths = mrmc.iterate(poi)

X_full = np.array(preprocessor.transform(adult_train.drop('Y', axis=1)))
Y_full = np.array(adult_train['Y'])

paths_processed = []
for path in paths:
    paths_processed.append(preprocessor.transform(path).to_numpy())

display = Display2DPaths(X_full, Y_full, title="UCI Adult Income Demo")
fig, ax = display.do_pca().set_clusters(km.cluster_centers_).set_paths(paths_processed).heatmap()
plt.show()
plt.close()


print("Run preprocessed MRMC")
mrm = MRM(alpha=alpha, ignore_negatives=True, 
          immutable_features=immutable_features, perturb_dir=perturb_dir, 
          preprocessor=None)

mrmc = MRMCIterator(n_clusters,
                    mrm,
                    max_iterations=40)

processed_adult = preprocessor.transform(filtered_adult)
processed_poi = preprocessor.transform(poi)

# Generate recourse paths
mrmc.fit(processed_adult, cluster_assignments)
print("Generating paths...")
paths = mrmc.iterate(processed_poi)
processed_paths = []
for path in paths:
    processed_paths.append(path.to_numpy())

X_full = np.array(preprocessor.transform(adult_train.drop('Y', axis=1)))
Y_full = np.array(adult_train['Y'])

display = Display2DPaths(X_full, Y_full, title="UCI Adult Income Demo")
fig, ax = display.do_pca().set_clusters(km.cluster_centers_).set_paths(processed_paths).heatmap()
plt.show()
plt.close()

# Show results
# utils.display_heatmap(X, Y, paths, clusters=km.cluster_centers_, do_pca=True, title="Paths to Each Cluster", size='large')