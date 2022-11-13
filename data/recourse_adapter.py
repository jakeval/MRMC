from typing import Any, Sequence
import pandas as pd
import abc


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


class RecourseAdapter(abc.ABC):
    """An abstract base class for recourse adapters.

    The adapter translates between the original (possibly categorical) data
    space and continuous embedded space where recourse directions are
    generated. It also iterates recourse by converting directions in embedded
    space to human-readable instructions and interpreting recourse instructions
    as a hypothetical user would.
    """

    @abc.abstractmethod
    def get_label(self) -> str:
        """Gets the dataset's label column name."""
        pass

    @abc.abstractmethod
    def transform(self, dataset: pd.DataFrame) -> EmbeddedDataFrame:
        """Transforms data from human-readable format to an embedded continuous
        space.

        Args:
            dataset: The data to transform.

        Returns:
            Transformed data.
        """

    @abc.abstractmethod
    def inverse_transform(self, dataset: EmbeddedDataFrame) -> pd.DataFrame:
        """Transforms data from an embedded continuous space to its original
        human-readable format.

        Args:
            dataset: The data to inverse transform.

        Returns:
            Inverse transformed data.
        """

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

    def transform_series(self, poi: pd.Series) -> EmbeddedSeries:
        """Transforms data from human-readable format to an embedded continuous
        space.

        Args:
            dataset: The data to transform.

        Returns:
            Transformed data.
        """
        return self.transform(poi.to_frame().T).iloc[0]

    def inverse_transform_series(self, poi: EmbeddedSeries) -> pd.Series:
        """Transforms data from an embedded continuous space to its original
        human-readable format.
        Args:
            dataset: The data to inverse transform.

        Returns:
            Inverse transformed data.
        """
        return self.inverse_transform(poi.to_frame().T).iloc[0]
