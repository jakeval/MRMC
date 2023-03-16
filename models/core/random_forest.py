from typing import Optional
from sklearn import ensemble, model_selection
import pandas as pd
import numpy as np
import joblib
import os
import json
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
    "n_estimators": [125, 250, 500, 1000],
    "max_features": [None, "sqrt"],
    "random_state": [model_constants.RANDOM_SEED],
    "min_samples_split": [0.02],
    "max_depth": [10, 15, 20],
    "ccp_alpha": [0.001, 0.005, 0.01],
}


class RandomForest(model_trainer.ModelTrainer):
    """A class for training, saving, and loading Random Forest models.

    Uses the SKLearn RandomForest class."""

    def __init__(
        self,
        dataset_name: data_loader.DatasetName,
        model_name: model_constants.ModelName,
        max_gridsearch_iterations: Optional[int] = None,
    ):
        """Creates a new LogisticRegression class.

        Args:
            dataset_name: The name of the dataset to load.
            model_name: The name of the model to train.
            max_gridsearch_iterations: An optional limit on the number of
                hyperparameter combinations to try during gridsearch."""
        super().__init__(
            model_type=model_constants.ModelType.RANDOM_FOREST,
            dataset_name=dataset_name,
            model_name=model_name,
        )
        self.max_gridsearch_iterations = max_gridsearch_iterations

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
        params_list = []

        all_params = list(model_selection.ParameterGrid(TRAINING_PARAMS))
        if self.max_gridsearch_iterations:
            max_iterations = self.max_gridsearch_iterations  # for convenience
            print(f"Running {max_iterations} of {len(all_params)}")
            all_params = all_params[: self.max_gridsearch_iterations]
        for i, params in enumerate(all_params):
            print("run", i)
            print(params)
            start_time = time.time()
            rf = ensemble.RandomForestClassifier(**params)
            rf.fit(training_data, training_labels)
            model = model_interface.SKLearnModel(
                rf, adapter, hyperparams=params
            )
            y_pred = model.predict(val_data)
            y_true = val_data[dataset_info.label_column]
            accuracy = (y_pred == y_true).mean()
            accuracies.append(accuracy)
            models.append(model)
            runtime = time.time() - start_time
            runtimes.append(runtime)
            params_list.append(params)

        for accuracy, runtime in zip(accuracies, runtimes):
            print(f"accuracy: {accuracy:.4f}, runtime: {runtime:.4f}")

        best_model = models[np.argmax(accuracies)]
        print("Selected params:")
        for key, value in params.items():
            print(f"{key}: {value}")
        return best_model

    def save_model(self, model: model_interface.Model, model_dir: str) -> None:
        """Saves a trained model to local disk.

        Saving is done with joblib.

        Args:
            model: The model to save.
            model_dir: The directory to save the model under."""
        super().save_model(model, model_dir)
        joblib.dump(model, os.path.join(model_dir, MODEL_FILENAME))
        with open(os.path.join(model_dir, "model_config.json"), "w") as f:
            json.dump(model.hyperparams, f)

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
