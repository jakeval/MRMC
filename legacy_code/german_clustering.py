import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from data import data_adapter as da
import matplotlib.pyplot as plt


dftrain, dftest, preprocessor = da.load_german_credit_dataset()

X = preprocessor.transform(dftrain[dftrain['Y'] == 1].drop('Y', axis=1))
print(X.shape)

losses = []

ks = list(range(1,20))
for k in ks:
    km = KMeans(n_clusters=k)
    km.fit(X)
    losses.append(-km.score(X))

plt.plot(ks, losses)
plt.show()
plt.close()
