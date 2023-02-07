from __future__ import annotations
import numpy as np
import dice_ml
from dice_ml import constants

from models import model_interface
from data import recourse_adapter
from sklearn import pipeline
from sklearn import base as sklearn_base


class ToySKLearn(
    sklearn_base.BaseEstimator,
    sklearn_base.ClassifierMixin,
):
    """Adapts the ToyModel to satisfy the SKLearn interface for use by
    DICE."""

    def __init__(self, model: ToyModel):
        self.model = model

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.model._predict(X)

    def predict_proba(self, X):
        pos_proba = self.model._predict_pos_proba(X)
        proba = np.zeros((pos_proba.shape[0], 2))
        proba[:, 0] = 1 - pos_proba
        proba[:, 1] = pos_proba
        return proba


class ToyModel(model_interface.Model):
    """A classifier for the synthetic data defined in
    confidence_checks/toy_data.py.

    It predicts a linear probability gradient from (-1, 0) to (1, 2). The
    probability function has constant value outside of this box.
    """

    def __init__(self, adapter: recourse_adapter.RecourseAdapter):
        super().__init__(adapter=adapter)

    def _predict_pos_proba(
        self, dataset: recourse_adapter.EmbeddedDataFrame
    ) -> np.ndarray:
        """Predicts a probability gradient from (-1, 0) to (1, 2)"""
        x = dataset.x.to_numpy()
        y = dataset.y.to_numpy()
        output = np.empty((x.shape[0], 2))
        output[:, 0] = (x + 1) / 2
        output[:, 1] = y / 2
        return np.minimum(np.maximum(output, 0), 1).mean(axis=1)

    def _predict(
        self, dataset: recourse_adapter.EmbeddedDataFrame
    ) -> np.ndarray:
        proba = self._predict_pos_proba(dataset)
        return np.where(proba > 0.5, 1, -1)

    def to_dice_model(self):
        fake_sklearn_model = ToySKLearn(self)
        model_pipeline = pipeline.Pipeline(
            steps=[
                ("adapter", self.adapter),
                ("classifier", fake_sklearn_model),
            ]
        )

        return dice_ml.Model(
            model=model_pipeline, backend=constants.BackEndTypes.Sklearn
        )
