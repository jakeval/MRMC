import numpy as np
import pandas as pd


class SyntheticModel:
    def __init__(self, cutoff, width):
        self.cutoff = cutoff
        self.width = width

    def fit(self, X, y):
        pass

    def predict(self, dataset):
        pass

    def predict_proba(self, dataset):
        X = dataset
        if type(dataset) == pd.DataFrame:
            X = dataset.to_numpy()
        y_coords = X[:,1]
        certainties = 0.5 + (y_coords - self.cutoff) / (2*self.width)
        return certainties.clip(0, 1)
