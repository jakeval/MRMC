from __future__ import annotations
from typing import Any, Sequence, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
import abc


# TODO(@jakeval): Reconsider this class's responsibilities -- it does too much.
# It should only translate between embedded and native data formats. Generating
# instructions and taking actions should be done separately.


class EmbeddedDataFrame(pd.DataFrame):
    """A wrapper around DataFrame for continuous data.

    Recourse directions are generated in embedded continuous space. DataFrames
    with categorical directions must be converted to EmbeddedDataFrames before
    directional recourse can be generated.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._check_types_are_numeric()

    def _check_types_are_numeric(self):
        for col, dtype in zip(self.columns, self.dtypes):
            if not pd.api.types.is_numeric_dtype(dtype):
                raise ValueError(
                    f"Column {col} has type {dtype} which is not numeric."
                )
        return True


class EmbeddedSeries(pd.Series):
    """A wrapper around Series for continuous data.

    Recourse directions are generated in embedded continuous space. Series
    with categorical directions must be converted to EmbeddedSeries before
    directional recourse can be generated.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._check_types_are_numeric()

    def _check_types_are_numeric(self):
        if not pd.api.types.is_numeric_dtype(self.dtype):
            raise ValueError(
                f"Series has type {self.dtype} which is not numeric."
            )
        return True


@dataclass
class RecourseAdapter(abc.ABC):
    """An abstract base class for recourse adapters.

    The adapter translates between the original (possibly categorical) data
    space and continuous embedded space where recourse directions are
    generated. It also iterates recourse by converting directions in embedded
    space to human-readable instructions and interpreting recourse instructions
    as a hypothetical user would.

    The RecourseAdapter also transforms a dataset's label column to -1/1
    encoding where -1 is the negative class and 1 is the positive class.

    Attributes:
        label_name: The name of the label feature.
        positive_label: The value of the label for the positive class.
        negative_label: The value of the label for the positive class. This is
            inferred automatically when the adapter's .fit() function is
            called.
    """

    label_name: str
    positive_label: Any
    negative_label: Optional[Any] = None

    @abc.abstractmethod
    def transform(self, dataset: pd.DataFrame) -> EmbeddedDataFrame:
        """Transforms data from human-readable format to an embedded continuous
        space.

        The label column is always encoded so that the positive outcome is 1
        and the negative outcome is -1.

        Args:
            dataset: The data to transform.

        Returns:
            Transformed data.
        """
        df = dataset.copy()
        if self.label_name in df.columns:
            df[self.label_name] = self.transform_label(df[self.label_name])
        return df

    @abc.abstractmethod
    def inverse_transform(self, dataset: EmbeddedDataFrame) -> pd.DataFrame:
        """Transforms data from an embedded continuous space to its original
        human-readable format.

        Args:
            dataset: The data to inverse transform.

        Returns:
            Inverse transformed data.
        """
        df = dataset.copy()
        if self.label_name in df.columns:
            df[self.label_name] = self.inverse_transform_label(
                df[self.label_name]
            )
        return df

    @abc.abstractmethod
    def directions_to_instructions(self, directions: EmbeddedSeries) -> Any:
        """Converts a direction in embedded space to a human-readable
        instructions format.

        Args:
            directions: The continuous recourse directions to convert.

        Returns:
            Human-readable instructions describing the recourse directions.
        """

    @abc.abstractmethod
    def interpret_instructions(
        self, poi: pd.Series, instructions: Any
    ) -> pd.Series:
        """Returns a new human-readable data point from an original POI and set
        of recourse instructions.
        Interprets the instructions by translating the original human-readable
        POI.

        Args:
            poi: The point of interest (POI) to translate.
            instructions: The recourse instructions to interpret.

        Returns:
            A new POI translated from the original by the recourse
            instructions.
        """

    @abc.abstractmethod
    def column_names(self, drop_label=True) -> Sequence[str]:
        """Returns the column names of the human-readable data.

        Args:
            drop_label: Whether the label column should be excluded in the
                output.

        Returns:
            A list of the column names.
        """

    @abc.abstractmethod
    def embedded_column_names(self, drop_label=True) -> Sequence[str]:
        """Returns the column names of the data in embedded continuous space.

        Args:
            drop_label: Whether the label column should be excluded in the
                output.
        Returns:
            A list of the column names.
        """

    def fit(self, dataset: pd.DataFrame) -> RecourseAdapter:
        """Fits the adapter to a dataset.

        This enables automatic encoding and decoding of the label column.

        Args:
            dataset: The dataset to fit the RecourseAdapter to.

        Returns:
            Itself. Fitting is done mutably.
        """
        labels = dataset[self.label_name]
        self.negative_label = labels[labels != self.positive_label].iloc[0]
        return self

    def transform_label(self, labels: pd.Series) -> EmbeddedSeries:
        """Encodes human-readable labels as a -1/1 encoded series.

        1 corresponds to the positive class and -1 corresponds to the
        negative class.

        Args:
            labels: The labels to encode.

        Returns:
            A series of the same length as labels with values 1 and -1.
        """
        y = np.where(labels == self.positive_label, 1, -1)
        return pd.Series(y, index=labels.index)

    def inverse_transform_label(self, y: EmbeddedSeries) -> pd.Series:
        """Transforms -1/1 encoded labels to their original human-readable
        format.

        1 corresponds to the positive class and -1 corresponds to the
        negative class.

        Args:
            y: The encoded labels to decode.

        Returns:
            A series of the same length as y with its original human-readable
            values.
        """
        labels = np.where(y == 1, self.positive_label, self.negative_label)
        return pd.Series(labels, index=y.index)

    def transform_series(self, data_series: pd.Series) -> EmbeddedSeries:
        """Transforms data from human-readable format to an embedded continuous
        space.

        Args:
            data_series: The data to transform.

        Returns:
            Transformed data.
        """
        return self.transform(data_series.to_frame().T).iloc[0]

    def inverse_transform_series(
        self, data_series: EmbeddedSeries
    ) -> pd.Series:
        """Transforms data from an embedded continuous space to its original
        human-readable format.
        Args:
            data_series: The data to inverse transform.

        Returns:
            Inverse transformed data.
        """
        return self.inverse_transform(data_series.to_frame().T).iloc[0]
