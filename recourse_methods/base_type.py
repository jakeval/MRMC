import abc
from typing import Sequence, Any
import pandas as pd


class RecourseMethod(abc.ABC):
    """An abstract base class for recourse methods."""

    @abc.abstractmethod
    def get_all_recourse_instructions(self, poi: pd.Series) -> Sequence[Any]:
        """Generates different recourse instructions for the poi for each of the k_directions."""

    @abc.abstractmethod
    def get_kth_recourse_instructions(self, poi: pd.Series, dir_index: int) -> Any:
        """Generates a single set of recourse instructions for the kth direction."""
