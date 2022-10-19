from __future__ import annotations
from data.adapters import adapter_types
from typing import Sequence, Optional, Mapping
from core import utils
from sklearn.preprocessing import StandardScaler
import pandas as pd


class StandardizingAdapter(adapter_types.RecourseAdapter):
    """A recourse adapter which standardizes continuous data to have mean 0 and
    standard deviation 1.

    The adapter also optionally simulates rescaling the recourse or adding
    random noise while interpreting recourse instructions."""
    def __init__(self,
                 perturb_ratio: Optional[float] = None,
                 rescale_ratio: Optional[float] = None,
                 label='Y'):
        """Creates a new ContinuousAdapter.

        Args:
            perturb_ratio: The magnitude of random noise relative to the
                recourse directions to add while interpreting recourse
                instructions.
            rescale_ratio: The amount to rescale the recourse directions by
                while interpreting recourse instructions."""
        self.label = label

        self.sc_dict: Mapping[str, StandardScaler] = None
        self.columns = None
        self.continuous_features = None
        self.perturb_ratio = perturb_ratio
        self.rescale_ratio = rescale_ratio

    def get_label(self) -> str:
        """Gets the dataset's label column name."""
        return self.label

    def fit(self, dataset: pd.DataFrame) -> StandardizingAdapter:
        """Fits the adapter to a dataset.

        Args:
            dataset: The data to fit.

        Returns:
            Itself. Fitting is done mutably."""
        self.columns = dataset.columns
        self.continuous_features = dataset.columns.difference([self.label])
        self.sc_dict = {}
        for feature in self.continuous_features:
            sc = StandardScaler()
            sc.fit(dataset[[feature]])
            self.sc_dict[feature] = sc
        return self

    def transform(self, dataset: pd.DataFrame) -> adapter_types.EmbeddedDataFrame:
        """Transforms data from human-readable format to an embedded continuous
        space by standardizing the data to have 0 mean and standard deviation 1.
        
        Args:
            dataset: The data to transform.
            
        Returns:
            Transformed data."""
        df = dataset.copy()
        for feature in self.continuous_features:
            if feature in df.columns:
                df[feature] = self.sc_dict[feature].transform(df[[feature]])
        return df

    def inverse_transform(self, dataset: adapter_types.EmbeddedDataFrame) -> pd.DataFrame:
        """Transforms data from an embedded continuous space to its original
        human-readable format by restoring the original dataset mean and
        standard deviation.

        Args:
            dataset: The data to inverse transform.

        Returns:
            Inverse transformed data."""
        df = dataset.copy()
        for feature in self.continuous_features:
            if feature in df.columns:
                df[feature] = self.sc_dict[feature].inverse_transform(df[[feature]])
        return df

    def directions_to_instructions(self, directions: adapter_types.EmbeddedSeries) -> adapter_types.EmbeddedSeries:
        """Converts a direction in embedded space to a human-readable instructions format.

        For this class, it is a no-op.
        
        Args:
            directions: The continuous recourse directions to convert.
            
        Returns:
            Human-readable instructions describing the recourse directions."""
        return directions

    def interpret_instructions(self, poi: pd.Series, instructions: adapter_types.EmbeddedSeries) -> pd.Series:
        """Returns a new human-readable data point from an original Point of
        Interest (POI) and set of recourse instructions.

        Converts the POI to the embedded space and translates it using the
        embedded space instructions. Then reconverts the POI to its original
        data space.

        If self.perturb_ratio is not None, adds random noise while translating
        the POI.

        If self.rescale_ratio is not None, rescales the magnitude of the
        translation.

        Args:
            poi: The point of interest (POI) to translate.
            instructions: The recourse instructions to interpret.

        Returns:
            A new POI translated from the original by the recourse instructions."""
        if self.perturb_ratio:
            instructions = utils.randomly_perturb_dir(
                instructions, self.perturb_ratio)
        if self.rescale_ratio:
            instructions = utils.rescale_dir(instructions, self.rescale_ratio)
        poi = self.transform_series(poi)
        cfe = poi + instructions
        return self.inverse_transform_series(cfe)

    def column_names(self, drop_label=True) -> Sequence[str]:
        """Returns the column names of the human-readable data.

        Args:
            drop_label: Whether the label column should be excluded in the
                output.

        Returns:
            A list of the column names."""
        if drop_label:
            return self.columns.difference([self.label])
        else:
            return self.columns

    def embedded_column_names(self, drop_label=True) -> Sequence[str]:
        """Returns the column names of the data in embedded continuous space.

        Args:
            drop_label: Whether the label column should be excluded in the
                output.

        Returns:
            A list of the column names."""
        if drop_label:
            return self.columns.difference([self.label])
        else:
            return self.columns
