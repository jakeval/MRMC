from __future__ import annotations
from typing import Any
import pandas as pd
import abc


class EmbeddedDataFrame(pd.DataFrame):
    """A wrapper around DataFrame for purely numeric data."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._check_types_are_numeric()

    def _check_types_are_numeric(self):
        for col, dtype in zip(self.columns, self.dtypes):
            if not pd.api.types.is_numeric_dtype(dtype):
                raise ValueError(f"Column {col} has type {dtype} which is not numeric.")
        return True


class EmbeddedSeries(pd.Series):
    """A wrapper around Series for purely numeric data."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._check_types_are_numeric()

    def _check_types_are_numeric(self):
        if not pd.api.types.is_numeric_dtype(self.dtype):
            raise ValueError(f"Series has type {self.dtype} which is not numeric.")
        return True


class Preprocessor(abc.ABC):
    """An abstract base class for dataset preprocessors.
    
    The preprocessor translates between human-readable and embedded space.
    """
    @abc.abstractmethod
    def fit(self, dataset: pd.DataFrame) -> Preprocessor:
        pass

    @abc.abstractmethod
    def transform(self, dataset: pd.DataFrame) -> EmbeddedDataFrame:
        """Transforms data from human-readable format to an embedded numeric space."""
        pass

    @abc.abstractmethod
    def inverse_transform(self, dataset: EmbeddedDataFrame) -> pd.DataFrame:
        """Transforms data from an embedded numeric space to its original human-readable format."""
        pass

    @abc.abstractmethod
    def directions_to_instructions(self, directions: EmbeddedSeries) -> Any:
        """Converts a direction in embedded space to some human-readable instructions format."""
        pass

    @abc.abstractmethod
    def interpret_instructions(self, poi: pd.Series, instructions: Any) -> pd.Series:
        """Returns a new human-readable data point from an original POI and set of instructions.
        
        Interprets the instructions by translating the original human-readable POI."""
        pass

    def transform_series(self, poi: pd.Series) -> EmbeddedSeries:
        """Transforms data from human-readable format to an embedded numeric space."""
        return self.transform(poi.to_frame().T).iloc[0]

    def inverse_transform_series(self, poi: EmbeddedSeries) -> pd.Series:
        """Transforms data from an embedded numeric space to its original human-readable format."""
        return self.inverse_transform(poi.to_frame().T).iloc[0]
