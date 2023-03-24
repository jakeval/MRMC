import abc
import os
from typing import Tuple, Mapping, Any, Optional
import json
import pandas as pd
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
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        dataset_info: base_loader.DatasetInfo,
    ) -> model_interface.Model:
        """Trains a model on the given training dataset.

        Evaluation on a test dataset is performed separately.

        Args:
            train_data: The training dataset to use.
            val_data: The validation dataset to use.
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
        train_data, val_data, test_data, dataset_info = data_loader.load_data(
            self.dataset_name, **(data_loader_kwargs or {})
        )
        model = self.train_model(train_data, val_data, dataset_info)
        results = self.evaluate_model(
            model, train_data, val_data, test_data, dataset_info
        )
        model_dir = self._get_model_dir()
        self.save_model(model, model_dir)
        self.save_results(results, model_dir)
        return model, results

    def evaluate_model(
        self,
        model: model_interface.Model,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        dataset_info: base_loader.DatasetInfo,
    ) -> Mapping[str, Mapping[str, float]]:
        """Evaluates a trained model.

        The results have format
        {
            "train": {
                "true_positives": int,
                "false_positives": int,
                "false_negatives": int,
                "accuracy": float
            },
            "val": {...},
            "test": {...}
        }

        Args:
            model: The model to evaluate.
            train_data: The training dataset.
            val_data: The validation dataset.
            test_data: The testing dataset.
            dataset_info: Information about the dataset.

        Returns:
            A dictionary containing results as described above."""

        def _evaluate_on_split(
            data: pd.DataFrame,
        ) -> Mapping[str, pd.DataFrame]:
            pos_mask = (
                data[dataset_info.label_column] == dataset_info.positive_label
            )
            y_pred = model.predict(data)
            y_true = data[dataset_info.label_column]

            results = {}
            results["accuracy"] = float((y_pred == y_true).mean())
            results["true_positives"] = int(
                (y_pred[pos_mask] == y_true[pos_mask]).sum()
            )
            results["false_positives"] = int(
                (y_pred[~pos_mask] != y_true[~pos_mask]).sum()
            )
            results["true_negatives"] = int(
                (y_pred[~pos_mask] == y_true[~pos_mask]).sum()
            )
            results["false_negatives"] = int(
                (y_pred[pos_mask] != y_true[pos_mask]).sum()
            )

            return results

        results = {}
        results["train"] = _evaluate_on_split(train_data)
        results["val"] = _evaluate_on_split(val_data)
        results["test"] = _evaluate_on_split(test_data)
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
