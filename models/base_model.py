import abc
import numpy as np


class BaseModel(abc.ABC):
    """A model interface type for any model defining the function
    predict_proba() as used in most sklearn models."""

    @abc.abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Given unlabeled dataset X, return an array of model certainties.

        The returned array is N by C where N is the number of datapoints and C
        is the number of classes.
        
        Args:
          X: A numpy array of size N by D.
          
        Returns:
          A numpy array of size N by C."""