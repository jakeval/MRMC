import numpy as np
from data import data_adapter as da
import timeit
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import pandas as pd

"""
This script was used to find the KDE hyperparameters on the datasets.

It tests both accuracy and prediction time.
"""


df, _, p = da.load_german_credit_dataset()
X = p.transform(df).drop('Y', axis=1).to_numpy()


def describe_kde(bandwidth):
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(X)
    scores = kde.score_samples(X)
    print(scores.mean(), scores.min(), scores.max())
    return scores.mean()


def check_kde(bandwidths, kernel, rtol=None):
    params = {
        'bandwidth': bandwidths,
        'kernel': [kernel],
    }
    if rtol is not None:
        params.update({'rtol': [rtol]})
    grid = GridSearchCV(KernelDensity(), params, cv=5)
    print("fit")
    grid.fit(X)
    df = pd.DataFrame(grid.cv_results_)
    print(df[['param_kernel', 'param_bandwidth', 'mean_test_score', 'mean_score_time', 'rank_test_score']])


if __name__ == '__main__':
    print("Start Selection")
    check_kde(np.linspace(0.1, 0.5, 20), 'gaussian')
    num_timing_samples = 5
    bandwidth = 0.29
    t1 = timeit.timeit(lambda: describe_kde(), number=num_timing_samples)
    print(t1)
