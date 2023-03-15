from sklearn import ensemble, model_selection
import pandas as pd
import numpy as np
import joblib
import os
import time
from data.datasets import base_loader
from data import data_loader
from data.adapters import continuous_adapter, categorical_adapter
from data import recourse_adapter
from models.core import model_trainer
from models import model_constants, model_interface


MODEL_FILENAME = "model.pkl"  # The default filename for saved LR models.
TRAINING_PARAMS = {
    "class_weight": ["balanced"],
    "n_estimators": [2**7, 2**9, 2**11],
    "max_features": [1, None, "sqrt"],
    "random_state": [model_constants.RANDOM_SEED],
    "min_samples_split": [2],  # [2, 4, 8, 16],
    "max_depth": [None],  # [4, 16, 64, None],
}


class RandomForest(model_trainer.ModelTrainer):
    """A class for training, saving, and loading Random Forest models.

    Uses the SKLearn RandomForest class."""

    def __init__(
        self,
        dataset_name: data_loader.DatasetName,
        model_name: model_constants.ModelName,
    ):
        """Creates a new LogisticRegression class.

        Args:
            dataset_name: The name of the dataset to load.
            model_name: The name of the model to train."""
        super().__init__(
            model_type=model_constants.ModelType.RANDOM_FOREST,
            dataset_name=dataset_name,
            model_name=model_name,
        )

    def train_model(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        dataset_info: base_loader.DatasetInfo,
    ) -> model_interface.Model:
        """Trains a model on the given training dataset.

        Args:
            train_data: The training dataset to use.
            val_data: The validation dataset to use.
            dataset_info: Information on the training dataset.

        Returns:
            A trained Model."""
        adapter = self._get_adapter(train_data, dataset_info)
        dataset = adapter.transform(train_data)
        training_data = dataset.drop(dataset_info.label_column, axis=1)
        training_labels = dataset[dataset_info.label_column]

        models = []
        accuracies = []
        runtimes = []

        for params in model_selection.ParameterGrid(TRAINING_PARAMS):
            start_time = time.time()
            rf = ensemble.RandomForestClassifier(**params)
            rf.fit(training_data, training_labels)
            model = model_interface.SKLearnModel(rf, adapter)
            y_pred = model.predict(val_data)
            y_true = val_data[dataset_info.label_column]
            accuracy = (y_pred == y_true).mean()
            accuracies.append(accuracy)
            models.append(model)
            runtime = time.time() - start_time
            runtimes.append(runtime)

        for accuracy, runtime in zip(accuracies, runtimes):
            print(f"accuracy: {accuracy:.4f}, runtime: {runtime:.4f}")

        best_model = models[np.argmax(accuracies)]
        return best_model

    def save_model(self, model: model_interface.Model, model_dir: str) -> None:
        """Saves a trained model to local disk.

        Saving is done with joblib.

        Args:
            model: The model to save.
            model_dir: The directory to save the model under."""
        super().save_model(model, model_dir)
        joblib.dump(model, os.path.join(model_dir, MODEL_FILENAME))

    def _load_model(self, model_dir: str) -> model_interface.Model:
        """Loads a trained model from local disk."""
        return joblib.load(os.path.join(model_dir, MODEL_FILENAME))

    def _get_adapter(
        self, dataset: pd.DataFrame, dataset_info: base_loader.DatasetInfo
    ) -> recourse_adapter.RecourseAdapter:
        """Creates a RecourseAdapter based on the dataset info.

        Datasets containing ordinal features are not currently supported. If
        dataset the contains categorical features, returns a OneHotAdapter.
        Otherwise returns a StandardizingAdapter.

        Args:
            dataset: The dataset to get a RecourseAdapter for.
            dataset_info: Info about the dataset.

        Raises:
            NotImplementedError is the dataset contains ordinal features.

        Returns:
            A RecourseAdapter fitted to the given dataset."""
        if dataset_info.ordinal_features:
            raise NotImplementedError(
                "Datasets with ordinal features aren't supported."
            )
        if dataset_info.categorical_features or dataset_info.ordinal_features:
            adapter = categorical_adapter.OneHotAdapter(
                categorical_features=dataset_info.categorical_features,
                continuous_features=dataset_info.continuous_features,
                label_column=dataset_info.label_column,
                positive_label=dataset_info.positive_label,
            ).fit(dataset)
        else:
            adapter = continuous_adapter.StandardizingAdapter(
                label_column=dataset_info.label_column,
                positive_label=dataset_info.positive_label,
            ).fit(dataset)
        return adapter
