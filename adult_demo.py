
from core.mrmc import MRM, MRMCIterator
from core import utils
from data import data_adapter as da

import numpy as np
from sklearn.cluster import KMeans

print("Load the dataset...")
adult_train, adult_test, preprocessor = da.load_adult_income_dataset()
print("Shape before processing: ", adult_train.shape)

# Filtering for Immutability...
print("Select a random POI...")
poi = da.random_poi(adult_train, drop_label=True)
immutable_features = ['age', 'sex', 'race'] #, 'relationship', 'marital-status']
feature_tolerances = {'age': 5}
print(poi)

filtered_adult = da.filter_from_poi(adult_train, poi, immutable_features, feature_tolerances) # always retains columns
print("Shape after immutability filtering: ", filtered_adult.shape)

print("Do clustering...")
X = np.array(preprocessor.transform(filtered_adult.drop('Y', axis=1)))
Y = np.array(filtered_adult['Y'])
n_clusters=3
km = KMeans(n_clusters=n_clusters)
km.fit(X[Y == 1])
cluster_assignments = km.predict(X[Y == 1])

# Set alpha values
alpha = lambda dist: utils.cliff_alpha(dist, cutoff=0.5, degree=2)
alpha_neg = lambda dist: utils.cliff_alpha(dist, cutoff=0.5, degree=3)

perturb_dir = None

print("Run MRM...")
# Prepare MRMC
mrm = MRM(alpha=alpha, alpha_neg=alpha_neg, ignore_negatives=True, immutable_features=immutable_features, perturb_dir=perturb_dir, preprocessor=preprocessor)
mrm.fit(filtered_adult)
newpoint = mrm.transform(poi)
print(newpoint)

for column in poi.columns:
    print(f"{column}: {poi[column].iloc[0]} -> {newpoint[column].iloc[0]}")

