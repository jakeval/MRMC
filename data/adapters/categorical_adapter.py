from __future__ import annotations
from data import recourse_adapter
from typing import Sequence, Optional, Mapping
from core import utils
from sklearn import preprocessing
import pandas as pd


# TODO(@jakeval): Reduce code duplication between this file and
# continuous_adpater.py


class OneHotAdapter(recourse_adapter.RecourseAdapter):
    """A recourse adapter which one-hot encodes categorical variables and
    standardizes continuous data to have mean 0 and standard deviation 1.

    The adapter also optionally simulates rescaling the recourse or adding
    random noise while interpreting recourse instructions."""

    def __init__(
        self,
        categorical_features: Sequence[str],
        continuous_features: Sequence[str],
        perturb_ratio: Optional[float] = None,
        rescale_ratio: Optional[float] = None,
        label: str = "Y",
    ):
        """Creates a new OneHotAdapter.

        Args:
            categorical_features: The names of the categorical features.
            continuous_features: The names of the continuous features.
            perturb_ratio: The magnitude of random noise relative to the
                recourse directions to add while interpreting recourse
                instructions.
            rescale_ratio: The amount to rescale the recourse directions by
                while interpreting recourse instructions.
            label: The name of the class label feature."""
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.label = label

        self.standard_scaler_dict: Mapping[
            str, preprocessing.StandardScaler
        ] = None
        self.onehot_dict: Mapping[str, preprocessing.OneHotEncoder] = None
        self.columns = None
        self.perturb_ratio = perturb_ratio
        self.rescale_ratio = rescale_ratio

    def get_label(self) -> str:
        """Gets the dataset's label column name."""
        return self.label

    def fit(self, dataset: pd.DataFrame) -> OneHotAdapter:
        """Fits the adapter to a dataset.

        Args:
            dataset: The data to fit.

        Returns:
            Itself. Fitting is done mutably."""
        self.standard_scaler_dict = {}
        self.onehot_dict = {}
        self.columns = dataset.columns
        for feature in self.continuous_features:
            standard_scaler = preprocessing.StandardScaler()
            standard_scaler.fit(dataset[[feature]])
            self.standard_scaler_dict[feature] = standard_scaler
        for feature in self.categorical_features:
            onehot_encoder = preprocessing.OneHotEncoder()
            onehot_encoder.fit(dataset[[feature]])
            self.onehot_dict[feature] = onehot_encoder
        return self

    def transform(
        self, dataset: pd.DataFrame
    ) -> recourse_adapter.EmbeddedDataFrame:
        """Transforms data from human-readable format to an embedded continuous
        space by standardizing the data and one-hot encoding categorical vars.

        Args:
            dataset: The data to transform.

        Returns:
            Transformed data."""
        df = dataset.copy()
        for feature in self.continuous_features:
            if feature in df.columns:
                df[feature] = self.standard_scaler_dict[feature].transform(
                    df[[feature]]
                )
        for feature in self.categorical_features:
            if feature in df.columns:
                onehot_encoder = self.onehot_dict[feature]
                feature_columns = onehot_encoder.get_feature_names_out(
                    [feature]
                )
                df[feature_columns] = onehot_encoder.transform(
                    df[[feature]]
                ).toarray()
                df = df.drop(feature, axis=1)
        return df

    def inverse_transform(
        self, dataset: recourse_adapter.EmbeddedDataFrame
    ) -> pd.DataFrame:
        """Transforms data from an embedded continuous space to its original
        human-readable format.

        Args:
            dataset: The data to inverse transform.

        Returns:
            Inverse transformed data."""
        df = dataset.copy()
        for feature in self.continuous_features:
            if feature in df.columns:
                df[feature] = self.standard_scaler_dict[
                    feature
                ].inverse_transform(df[[feature]])
        for feature in self.categorical_features:
            onehot_encoder = self.onehot_dict[feature]
            feature_columns = onehot_encoder.get_feature_names_out([feature])
            if df.columns.intersection(feature_columns).any():
                df[feature] = onehot_encoder.inverse_transform(
                    df[feature_columns]
                )
                df = df.drop(feature_columns, axis=1)
        return df

    def directions_to_instructions(
        self, directions: recourse_adapter.EmbeddedSeries
    ) -> recourse_adapter.EmbeddedSeries:
        """Converts a direction in embedded space to a human-readable
        instructions format.

        For this class, it is a no-op.

        Args:
            directions: The continuous recourse directions to convert.

        Returns:
            Human-readable instructions describing the recourse directions."""
        return directions

    def interpret_instructions(
        self, poi: pd.Series, instructions: recourse_adapter.EmbeddedSeries
    ) -> pd.Series:
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
            A new POI translated from the original by the recourse
            instructions."""
        if self.perturb_ratio:
            instructions = utils.randomly_perturb_dir(
                instructions, self.perturb_ratio
            )
        if self.rescale_ratio:
            instructions = utils.rescale_dir(instructions, self.rescale_ratio)
        poi = self.transform_series(poi)
        counterfactual = poi + instructions
        return self.inverse_transform_series(counterfactual)

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
        columns = self._get_feature_names_out(self.columns)
        if drop_label:
            return [column for column in columns if column != self.label]
        else:
            return columns

    def _get_feature_names_out(self, features: Sequence[str]) -> Sequence[str]:
        features_out = []
        for feature in features:
            if feature in self.categorical_features:
                features_out += list(
                    self.onehot_dict[feature].get_feature_names_out([feature])
                )
            else:
                features_out.append(feature)
        return features_out
