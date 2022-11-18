import abc
from typing import Any
import numpy as np
import pandas as pd
import dice_ml
from dice_ml import constants
from sklearn import pipeline


from dataclasses import dataclass
from data import recourse_adapter


@dataclass
class Model(abc.ABC):
    """A Model interface allowing MRMC experiments to use models from different
    backends.

    Each backend (sklearn, pytorch, etc) should subclass this class.

    The functions implemented are accessors to a model's class prediction and
    class probability prediction functions. The base Model class uses a
    RecourseAdapter to automatically drop the dataset's label column (if
    included) and convert to and from human-readable representations and
    continuous embedded representations of the data.

    Attributes:
        adapter: A RecourseAdapter used to convert between human-readable
            and continuous embedded data representations."""

    adapter: recourse_adapter.RecourseAdapter

    @abc.abstractmethod
    def _predict_pos_proba(
        self, dataset: recourse_adapter.EmbeddedDataFrame
    ) -> np.ndarray:
        """Predicts the probability of a positive outcome for each example in
        the dataset.

        Args:
            dataset: A dataset without the label column and pre-transformed
                using the class's RecourseAdapter.

        Returns:
            A 1-dimensional numpy array of positive class probabilities."""

    @abc.abstractmethod
    def _predict(
        self, dataset: recourse_adapter.EmbeddedDataFrame
    ) -> np.ndarray:
        """Predicts the class label for each example in the dataset.

        The class predictions should be in embedded space.

        Args:
            dataset: A dataset without the label column and pre-transformed
                using the class's RecourseAdapter.

        Returns:
            A 1-dimensional numpy array of class predictions."""

    @abc.abstractmethod
    def to_dice_model(self) -> dice_ml.Model:
        """Returns a dice_ml Model based on this model."""

    def predict_pos_proba(self, dataset: pd.DataFrame) -> pd.Series:
        """Predicts the probability of a positive outcome for each example in
        the dataset.

        If the dataset includes the label column, the column is automatically
        dropped. The dataset should be passed in its unprocessed human-readable
        format.

        Args:
            dataset: The dataset to predict over.

        Returns:
            A series where element i of the series is the probability of a
            positive outcome for element i of the dataset."""
        dataset = self.adapter.transform(dataset)
        if self.adapter.label_column in dataset.columns:
            dataset = dataset.drop(self.adapter.label_column, axis=1)
        proba = self._predict_pos_proba(dataset)
        return pd.Series(proba, index=dataset.index)

    def predict(self, dataset: pd.DataFrame) -> pd.Series:
        """Predicts the class for each example in the dataset.

        If the dataset includes the label column, the column is automatically
        dropped. The dataset should be passed in its unprocessed human-readable
        format.

        Args:
            dataset: The dataset to predict over.

        Returns:
            A series where element i of the series is the predicted
            human-readable class of example i of the dataset."""
        dataset = self.adapter.transform(dataset)
        if self.adapter.label_column in dataset.columns:
            dataset = dataset.drop(self.adapter.label_column, axis=1)
        y_pred = self._predict(dataset)
        y_pred_series = pd.Series(y_pred, index=dataset.index)
        return self.adapter.inverse_transform_label(y_pred_series)

    def predict_pos_proba_series(self, x: pd.Series) -> float:
        """Predicts the probability of a positive outcome for a single data
        example.

        If the example includes a label, the label is automatically dropped.
        The example should be passed in its unprocessed human-readable format.

        Args:
            x: The series to predict for.

        Returns:
            The probability that example x belongs to the positive class."""
        dataset = x.to_frame().T
        # The label is dropped in self.predict_pos_proba()
        return self.predict_pos_proba(dataset).iloc[0]

    def predict_series(self, x: pd.Series) -> Any:
        """Predicts the class label for a single data example.

        If the example includes a label, the label is automatically dropped.
        The example should be passed in its unprocessed human-readable format.

        Args:
            x: The series to predict for.

        Returns:
            The predicted class label of the given example."""
        dataset = x.to_frame().T
        # The label is dropped in self.predict()
        return self.predict(dataset).iloc[0]


class SKLearnModel(Model):
    """An implementation of the Model class for SKLearn models."""

    def __init__(
        self, sklearn_model, adapter: recourse_adapter.RecourseAdapter
    ):
        """Creates a new SKLearnModel.

        Args:
            sklearn_model: A prediction model from sklearn.
            adapter: The RecourseAdapter used to process data for the sklearn
                model."""
        super().__init__(adapter=adapter)
        self.model = sklearn_model
        self.pos_class_index = np.where(self.model.classes_ == 1)[0][0]

    def _predict_pos_proba(
        self, dataset: recourse_adapter.EmbeddedDataFrame
    ) -> np.ndarray:
        """Predicts the probability of a positive outcome for each example in
        the dataset by using the model.predict_proba() function.

        Args:
            dataset: A dataset without the label column and pre-transformed
                using the class's RecourseAdapter.

        Returns:
            A 1-dimensional numpy array of positive class probabilities."""
        return self.model.predict_proba(dataset)[:, self.pos_class_index]

    def _predict(
        self, dataset: recourse_adapter.EmbeddedDataFrame
    ) -> np.ndarray:
        """Predicts the class label for each example in the dataset by using
        the model.predict() function.

        Args:
            dataset: A dataset without the label column and pre-transformed
                using the class's RecourseAdapter.

        Returns:
            A 1-dimensional numpy array of class predictions."""
        return self.model.predict(dataset)

    def to_dice_model(self) -> dice_ml.Model:
        """Returns a dice_ml Model based on this model.

        This is done by bundling the sklearn model with the RecourseAdapter
        which preprocesses the data together in a Pipeline. A DICE model is
        then created using that Pipelin."""
        model_pipeline = pipeline.Pipeline(
            steps=[
                ("adapter", self.adapter),
                ("classifier", self.model),
            ]
        )

        return dice_ml.Model(
            model=model_pipeline, backend=constants.BackEndTypes.Sklearn
        )
