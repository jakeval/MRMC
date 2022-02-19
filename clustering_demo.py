from core.mrmc import MRM, MRMCIterator, MRMIterator
from core import utils
from data import data_adapter as da
from models import random_forest
from visualize.two_d_plots import Display2DPaths

from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


adult_train, adult_test, preprocessor = da.load_adult_income_dataset()

X = np.array(preprocessor.transform(adult_train.drop('Y', axis=1)))
Y = np.array(adult_train['Y'])

X_test = np.array(preprocessor.transform(adult_test.drop('Y', axis=1)))
Y_test = np.array(adult_test['Y'])

print("Train a model...")
model, accuracy = random_forest.train_model(X, Y, X_test, Y_test)

model_scores = model.predict_proba(X)
adult_train = da.filter_from_model(adult_train, model_scores)
print("Shape after accuracy filtering: ", adult_train.shape)



def do_clustering(df, df2):
    train_losses = []
    val_losses = []
    X = np.array(preprocessor.transform(df.drop('Y', axis=1)))
    Y = np.array(df['Y'])

    X2 = np.array(preprocessor.transform(df2.drop('Y', axis=1)))
    Y2 = np.array(df2['Y'])

    k = [1,2,3,4,5,6,7,8,9,10]
    for n_clusters in k:
        print(f"k={n_clusters}")
        km = KMeans(n_clusters=n_clusters)
        km.fit(X[Y == 1])
        # train_losses.append(km.inertia_)
        train_losses.append(km.score(X[Y == 1]))
        val_losses.append(km.score(X2[Y2 == 1]))
    
    print(train_losses)
    print(val_losses)

    return train_losses, val_losses, k

tl, vl, k = do_clustering(adult_train, adult_test)

plt.plot(k, tl, label='test loss')
plt.plot(k, vl, label='val loss')
plt.legend()

plt.show()
plt.close()