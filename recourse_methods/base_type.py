import abc
from typing import Sequence, Any
import pandas as pd
from data import recourse_adapter


class RecourseMethod(abc.ABC):
    """An abstract base class for recourse methods."""

    @abc.abstractmethod
    def get_all_recourse_directions(
        self, poi: recourse_adapter.EmbeddedSeries
    ) -> recourse_adapter.EmbeddedDataFrame:
        """Generates different recourse directions for the poi for each of the
        k_directions.

        Args:
            poi: The Point of Interest (POI) to find recourse directions for.

        Returns:
            A DataFrame containing recourse directions for the POI.
        """

    @abc.abstractmethod
    def get_all_recourse_instructions(self, poi: pd.Series) -> Sequence[Any]:
        """Generates different recourse instructions for the poi for each of
        the k_directions.

        Whereas recourse directions are vectors in embedded space,
        instructions are human-readable guides for how to follow those
        directions in the original data space.

        Args:
            poi: The Point of Interest (POI) to find recourse instructions for.

        Returns:
            A Sequence recourse instructions for the POI.
        """

    @abc.abstractmethod
    def get_kth_recourse_instructions(
        self, poi: pd.Series, dir_index: int
    ) -> Any:
        """Generates a single set of recourse instructions for the kth
        direction.

        Args:
            poi: The Point of Interest (POI) to get the kth recourse
            instruction for.

        Returns:
            Instructions for the POI to achieve the recourse.
        """
