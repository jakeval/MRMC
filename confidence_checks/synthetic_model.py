import numpy as np

from models import model_interface
from data import recourse_adapter


class SyntheticModel(model_interface.Model):
    """A classifier for the synthetic data defined in
    confidence_checks/synthetic_data.py.

    It predicts a linear probability gradient from (-1, 0) to (1, 2). The
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
        pass
