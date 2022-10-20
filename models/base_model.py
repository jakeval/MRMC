import abc
import numpy as np


"""
we will use different models
 - model type
 - model parameters
 - training dataset

all models satisfy same interface

models must be trained and saved

how to select and load the model?
 - dataset
 - type
 - name

what is saved with a model?
 - the actual model
 - the dataset, type, and name
 - the filename and github commit used to create it
 - the config file used to train it
 - the evaluation info

how to find a model?
 - check model_dir/type/dataset/name


model_dir/type/train_model.py
 - mainfile for training
 - contains model definition and training API
 - hyperparameter tuning, dataset split, evaluation, model saving

use models/training_utils.py

from models import model_loader
# use enums

how to train all models?
 - maybe not necessary

how to train a single model?
 - provide dataset, type, and name
 - auto find github commit
 - config file (preferably located already at model_dir/type/dataset/name)
 - 
"""


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
