import abc
import numpy as np
from dataclasses import dataclass

from data import recourse_adapter


#  TODO(@jakeval): Refactor this class
@dataclass
class Model(abc.ABC):
    """A model interface type for any model defining the function
    predict_proba()."""

    adapter: recourse_adapter.RecourseAdapter

    @abc.abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Given unlabeled dataset X, return an array of model certainties.

        The returned array is N by C where N is the number of datapoints and C
        is the number of classes."""
