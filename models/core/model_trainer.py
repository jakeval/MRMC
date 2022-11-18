import abc
import os
from typing import Tuple, Mapping, Any, Optional
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass
from models import model_constants, model_interface
from data import data_loader
from data.datasets import base_loader


SPLIT_RATIO = 0.7  # The default train/test split ratio.
RESULTS_NAME = "results.json"  # The name of the model evaulation results file.


@dataclass
class ModelTrainer(abc.ABC):
    """An abstract base class for training, saving, and loading ML models.

    Attributes:
        model_type: The model family type to train (logistic regression, random
            forest, etc).
        model_name: The name of the specific model to train (default, version1,
            etc).
        dataset_name: The name of the dataset to train the model on."""

    model_type: model_constants.ModelType
    model_name: model_constants.ModelName
    dataset_name: data_loader.DatasetName

    @abc.abstractmethod
    def train_model(
        self, dataset: pd.DataFrame, dataset_info: base_loader.DatasetInfo
    ) -> model_interface.Model:
        """Trains a model on the given training dataset.

        Evaluation on a test dataset is performed separately.

        Args:
            dataset: The training dataset to use.
            dataset_info: Information on the training dataset.

        Returns:
            A trained Model."""

    @abc.abstractmethod
    def save_model(self, model: model_interface.Model, model_dir: str) -> None:
        """Saves a trained model to local disk.

        Args:
            model: The trained model object to save.
            model_dir: The directory to save the model under."""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    @abc.abstractmethod
    def _load_model(self, model_dir: str) -> model_interface.Model:
        """Loads a trained model from local disk.

        Args:
            model_dir: The location of the saved model to load.

        Returns:
            The saved model."""

    def load_model(self):
        """Loads a trained model from local disk.

        Returns:
            The saved model."""
        return self._load_model(self._get_model_dir())

    def new_model(
        self, data_loader_kwargs: Optional[Mapping[str, Any]] = None
    ) -> Tuple[model_interface.Model, Mapping[str, float]]:
        """Creates, trains, evaluates, and saves a new Model.

        Data is loaded using the dataset_name and split using
        self.split_dataset(). The model is trained using the overridden
        self.train_model() and evaluated using self.evaluate_model(). Finally,
        the model and results are saved to local disk.

        By default, the model results are the training and test accuracy.

        Args:
            data_loader_kwargs: Key-word arguments to pass to the dataset
                loader. This is usually not needed. Possible key word arguments
                depend on the dataset being loaded and can be seen in the
                data/data_loader.py files.

        Returns:
            The trained model and a dictionary of results."""
        dataset, dataset_info = data_loader.load_data(
            self.dataset_name, **(data_loader_kwargs or {})
        )
        train, test = self.split_dataset(dataset)
        model = self.train_model(train, dataset_info)

        results = self.evaluate_model(model, train, test, dataset_info)
        model_dir = self._get_model_dir()
        self.save_model(model, model_dir)
        self.save_results(results, model_dir)
        return model, results

    def split_dataset(
        self, dataset: pd.DataFrame, split_ratio: float = SPLIT_RATIO
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Randomly splits the dataset.

        A random seed from model_constants.RANDOM_SEED is used.

        Args:
            dataset: The dataset to split.
            split_ratio: The ratio of training to test size.

        Returns:
            Training and test datasets."""
        rng = np.random.default_rng(seed=model_constants.RANDOM_SEED)
        num_train = int(np.floor(dataset.shape[0] * split_ratio))
        idx = np.arange(dataset.shape[0])
        rng.shuffle(idx)
        train = dataset.iloc[idx[:num_train]]
        test = dataset.iloc[idx[num_train:]]
        return train, test

    def evaluate_model(
        self,
        model: model_interface.Model,
        train: pd.DataFrame,
        test: pd.DataFrame,
        dataset_info: base_loader.DatasetInfo,
    ) -> Mapping[str, float]:
        """Evaluates a trained model.

        The results have format
        {
            "train": {
                "positive_accuracy": float,
                "negative_accuracy": float,
                "total_accuracy": float
            },
            "test": {
                "positive_accuracy": float,
                "negative_accuracy": float,
                "total_accuracy": float
            }
        }

        Args:
            model: The model to evaluate.
            train: The training dataset.
            test: The testing dataset.
            dataset_info: Information about the dataset.

        Returns:
            A dictionary containing results as described above."""
        results = {"train_accuracy": None, "test_accuracy": None}
        results = {
            "train": {
                "positive_accuracy": None,
                "negative_accuracy": None,
                "total_accuracy": None,
            },
            "test": {
                "positive_accuracy": None,
                "negative_accuracy": None,
                "total_accuracy": None,
            },
        }
        train_pos_mask = (
            train[dataset_info.label_name] == dataset_info.positive_label
        )
        test_pos_mask = (
            test[dataset_info.label_name] == dataset_info.positive_label
        )

        results["train"]["total_accuracy"] = self._get_accuracy(
            model.predict(train), train[dataset_info.label_name]
        )
        results["train"]["positive_accuracy"] = self._get_accuracy(
            model.predict(train[train_pos_mask]),
            train.loc[train_pos_mask, dataset_info.label_name],
        )
        results["train"]["negative_accuracy"] = self._get_accuracy(
            model.predict(train[~train_pos_mask]),
            train.loc[~train_pos_mask, dataset_info.label_name],
        )

        results["test"]["total_accuracy"] = self._get_accuracy(
            model.predict(test), test[dataset_info.label_name]
        )
        results["test"]["positive_accuracy"] = self._get_accuracy(
            model.predict(test[test_pos_mask]),
            test.loc[test_pos_mask, dataset_info.label_name],
        )
        results["test"]["negative_accuracy"] = self._get_accuracy(
            model.predict(test[~test_pos_mask]),
            test.loc[~test_pos_mask, dataset_info.label_name],
        )
        return results

    def save_results(
        self, results: Mapping[str, float], model_dir: str
    ) -> None:
        """Saves the results dictionary alongside the model.

        Results are saved as results.json.

        Args:
            model_dir: Where the trained model is saved.
            results: The results to save alongside the trained model."""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        results_filepath = os.path.join(model_dir, RESULTS_NAME)
        with open(results_filepath, "w") as f:
            json.dump(results, f)

    def _get_accuracy(self, y_pred: pd.Series, y_true: pd.Series) -> float:
        """Calculates accuracy from predicted and ground truth labels.

        Args:
            y_pred: The predicted labels.
            y_true: The ground truth labels.

        Returns:
            Model accuracy as a float."""
        return (y_pred == y_true).sum() / y_pred.size

    def _get_model_dir(self) -> str:
        """Formats the model directory.

        The model directory has format
        MODEL_DIR/model_type/dataset_name/model_name/"""
        return os.path.join(
            model_constants.MODEL_DIR,
            self.model_type.value,
            self.dataset_name.value,
            self.model_name.value,
        )
