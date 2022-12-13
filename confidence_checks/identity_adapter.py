from __future__ import annotations
from typing import Optional, Any, Sequence
import pandas as pd
from core import utils
from data import recourse_adapter


class IdentityAdapter(recourse_adapter.RecourseAdapter):
    """A recourse adapter used for confidence checking which passes through
    continuous data without altering it.

    The adapter also optionally simulates rescaling the recourse or adding
    random noise while interpreting recourse instructions.
    """

    def __init__(
        self,
        label_column: str,
        perturb_ratio: Optional[float] = None,
        rescale_ratio: Optional[float] = None,
        positive_label: Any = 1,
    ):
        super().__init__(
            label_column=label_column, positive_label=positive_label
        )
        self.columns = None
        self.perturb_ratio = perturb_ratio
        self.rescale_ratio = rescale_ratio

    def fit(self, dataset: pd.DataFrame) -> IdentityAdapter:
        super().fit(dataset)
        self.columns = dataset.columns
        return self

    def transform(
        self, dataset: pd.DataFrame
    ) -> recourse_adapter.EmbeddedDataFrame:
        df = super().transform(dataset)
        return df

    def inverse_transform(
        self, dataset: recourse_adapter.EmbeddedDataFrame
    ) -> pd.DataFrame:
        df = super().inverse_transform(dataset)
        return df

    def directions_to_instructions(
        self, directions: recourse_adapter.EmbeddedSeries
    ) -> recourse_adapter.EmbeddedSeries:
        return directions

    def interpret_instructions(
        self, poi: pd.Series, instructions: recourse_adapter.EmbeddedSeries
    ) -> pd.Series:
        """Interprets the recourse instructions by moving the POI in the
        direction.

        It may also perturb or rescale the direction followed.
        """
        if self.perturb_ratio:
            instructions = utils.randomly_perturb_direction(
                instructions, self.perturb_ratio
            )
        if self.rescale_ratio:
            instructions = instructions * self.rescale_ratio
        poi = self.transform_series(poi)
        counterfactual = poi + instructions
        return self.inverse_transform_series(counterfactual)

    def column_names(self, drop_label=True) -> Sequence[str]:
        if drop_label:
            return self.columns.difference([self.label_column])
        else:
            return self.columns

    def embedded_column_names(self, drop_label=True) -> Sequence[str]:
        if drop_label:
            return self.columns.difference([self.label_column])
        else:
            return self.columns
