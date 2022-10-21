from __future__ import annotations
from typing import Any, Sequence, Mapping, Optional
import pandas as pd
import abc
from core import utils
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


class RecourseAdapter(abc.ABC):
    """An abstract base class for dataset adapters.
    
    The adapter translates between human-readable and embedded space.
    """
    @abc.abstractmethod
    def get_label(self) -> str:
        """Gets the dataset's label column name."""

    @abc.abstractmethod
    def transform(self, dataset: pd.DataFrame) -> EmbeddedDataFrame:
        """Transforms data from human-readable format to an embedded numeric space."""

    @abc.abstractmethod
    def inverse_transform(self, dataset: EmbeddedDataFrame) -> pd.DataFrame:
        """Transforms data from an embedded numeric space to its original human-readable format."""

    @abc.abstractmethod
    def directions_to_instructions(self, directions: EmbeddedSeries) -> Any:
        """Converts a direction in embedded space to some human-readable instructions format."""

    @abc.abstractmethod
    def interpret_instructions(self, poi: pd.Series, instructions: Any) -> pd.Series:
        """Returns a new human-readable data point from an original POI and set of instructions.
        
        Interprets the instructions by translating the original human-readable POI."""

    @abc.abstractmethod
    def column_names(self, drop_label=True) -> Sequence[str]:
        """Returns the column names of the human-readable data."""

    @abc.abstractmethod
    def embedded_column_names(self, drop_label=True) -> Sequence[str]:
        """Returns the column names of the data in embedded numeric space."""

    def transform_series(self, poi: pd.Series) -> EmbeddedSeries:
        """Transforms data from human-readable format to an embedded numeric space."""
        return self.transform(poi.to_frame().T).iloc[0]

    def inverse_transform_series(self, poi: EmbeddedSeries) -> pd.Series:
        """Transforms data from an embedded numeric space to its original human-readable format."""
        return self.inverse_transform(poi.to_frame().T).iloc[0]


class NaiveAdapter(RecourseAdapter):
    """A Naive adapter which standardizes numeric columns and one hot encodes categorical columns."""
    def __init__(self,
                 categorical_features: Sequence[str],
                 continuous_features: Sequence[str],
                 perturb_ratio: Optional[float] = None,
                 rescale_ratio: Optional[float] = None,
                 label='Y'):
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.label = label

        self.sc_dict: Mapping[str, StandardScaler] = None
        self.ohe_dict: Mapping[str, OneHotEncoder] = None
        self.columns = None
        self.perturb_ratio = perturb_ratio
        self.rescale_ratio = rescale_ratio

    def get_label(self) -> str:
        return self.label

    def fit(self, dataset: pd.DataFrame) -> NaiveAdapter:
        self.sc_dict = {}
        self.ohe_dict = {}
        self.columns = dataset.columns
        for feature in self.continuous_features:
            sc = StandardScaler()
            sc.fit(dataset[[feature]])
            self.sc_dict[feature] = sc
        for feature in self.categorical_features:
            ohe = OneHotEncoder()
            ohe.fit(dataset[[feature]])
            self.ohe_dict[feature] = ohe
        return self

    def transform(self, dataset: pd.DataFrame) -> EmbeddedDataFrame:
        df = dataset.copy()
        for feature in self.continuous_features:
            if feature in df.columns:
                df[feature] = self.sc_dict[feature].transform(df[[feature]])
        for feature in self.categorical_features:
            if feature in df.columns:
                ohe = self.ohe_dict[feature]
                feature_columns = ohe.get_feature_names_out([feature])
                df[feature_columns] = ohe.transform(df[[feature]]).toarray()
                df = df.drop(feature, axis=1)
        return df

    def inverse_transform(self, dataset: EmbeddedDataFrame) -> pd.DataFrame:
        df = dataset.copy()
        for feature in self.continuous_features:
            if feature in df.columns:
                df[feature] = self.sc_dict[feature].inverse_transform(df[[feature]])
        for feature in self.categorical_features:
            ohe = self.ohe_dict[feature]
            feature_columns = ohe.get_feature_names_out([feature])
            if df.columns.intersection(feature_columns).any():
                df[feature] = ohe.inverse_transform(df[feature_columns])
                df = df.drop(feature_columns, axis=1)
        return df

    def directions_to_instructions(self, directions: EmbeddedSeries) -> EmbeddedSeries:
        return directions

    def interpret_instructions(self, poi: pd.Series, instructions: EmbeddedSeries) -> pd.Series:
        if self.perturb_ratio:
            instructions = utils.randomly_perturb_dir(instructions, self.perturb_ratio)
        if self.rescale_ratio:
            instructions = utils.rescale_dir(instructions, self.rescale_ratio)
        poi = self.transform_series(poi)
        cfe = poi + instructions
        return self.inverse_transform_series(cfe)

    def column_names(self, drop_label=True) -> Sequence[str]:
        if drop_label:
            return self.columns.difference([self.label])
        else:
            return self.columns

    def embedded_column_names(self, drop_label=True) -> Sequence[str]:
        columns = self._get_feature_names_out(self.columns)
        if drop_label:
            return [column for column in columns if column != self.label]
        else:
            return columns

    def _get_feature_names_out(self, features: Sequence[str]) -> Sequence[str]:
        features_out = []
        for feature in features:
            if feature in self.categorical_features:
                features_out += list(self.ohe_dict[feature].get_feature_names_out([feature]))
            else:
                features_out.append(feature)
        return features_out


class PassthroughAdapter(RecourseAdapter):
    def __init__(self, label='Y'):
        self.columns = None
        self.label = label

    def get_label(self) -> str:
        return self.label

    def fit(self, dataset: pd.DataFrame) -> PassthroughAdapter:
        self.columns = dataset.columns
        return self

    def transform(self, dataset: pd.DataFrame) -> EmbeddedDataFrame:
        return dataset

    def inverse_transform(self, dataset: EmbeddedDataFrame) -> pd.DataFrame:
        return dataset

    def directions_to_instructions(self, directions: EmbeddedSeries) -> EmbeddedSeries:
        return directions

    def interpret_instructions(self, poi: pd.Series, instructions: EmbeddedSeries) -> pd.Series:
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
