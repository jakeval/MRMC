from __future__ import annotations
from typing import Sequence
import pandas as pd
from data.adapters import adapter_types


class PassthroughPreprocessor(adapter_types.RecourseAdapter):
    def __init__(self, label='Y'):
        self.columns = None
        self.label = label

    def get_label(self) -> str:
        return self.label

    def fit(self, dataset: pd.DataFrame) -> PassthroughPreprocessor:
        """Fits the adapter to a dataset.
        
        Args:
            dataset: The data to fit.

        Returns:
            Itself. Fitting is done mutably."""
        self.columns = dataset.columns
        return self

    def transform(self, dataset: pd.DataFrame) -> adapter_types.EmbeddedDataFrame:
        return dataset

    def inverse_transform(self, dataset: adapter_types.EmbeddedDataFrame) -> pd.DataFrame:
        return dataset

    def directions_to_instructions(self, directions: adapter_types.EmbeddedSeries) -> adapter_types.EmbeddedSeries:
        return directions

    def interpret_instructions(self, poi: pd.Series, instructions: adapter_types.EmbeddedSeries) -> pd.Series:
        return poi + instructions

    def column_names(self, drop_label=True) -> Sequence[str]:
        if drop_label:
            return self.columns.difference([self.label])
        else:
            return self.columns

    def embedded_column_names(self, drop_label=True) -> Sequence[str]:
        if drop_label:
            return self.columns.difference([self.label])
        else:
            return self.columns
