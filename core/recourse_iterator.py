from typing import Protocol, Sequence
import pandas as pd
from recourse_methods.base_type import RecourseMethod
from models import base_model
from data import recourse_adapter


class CertaintyChecker(Protocol):
    """Returns the model certainty as a 1-dimensional array."""
    def check_certainty(poi: pd.Series) -> float:
        """Returns the model certainty as a 1-dimensional array."""
        pass


def wrap_model(model: base_model.BaseModel, positive_index: int = 1) -> CertaintyChecker:
    """Returns a CertaintyChecker function from an sklearn model."""
    def check_certainty(poi: pd.Series) -> float:
        proba = model.predict_proba(poi.to_frame().T.to_numpy())
        return proba[0,positive_index]
    return check_certainty


class RecourseIterator:
    """Class for iterating recourse."""
    def __init__(self,
                 recourse_method: RecourseMethod,
                 adapter: recourse_adapter.RecourseAdapter,
                 certainty_cutoff: float = None,
                 check_certainty: CertaintyChecker = None):
        """Creates a new recourse iterator.

        Args:
            recourse_method: The recourse method to use.
            adapter: A RecourseAdapter object to transform the data between
                human-readable and embedded space.
            certainty_cutoff: If not None, stop iterating early if the model certainty reaches the cutoff.
            check_certainty: If not None, the function used to compute model certainty.
        """
        self.recourse_method = recourse_method
        self.certainty_cutoff = certainty_cutoff
        self.check_certainty = check_certainty
        self.adapter = adapter

    def iterate_k_recourse_paths(self, poi: pd.Series, max_iterations: int) -> Sequence[pd.DataFrame]:
        """Generates one recourse path for each of the model recourse directions."""
        all_instructions = self.recourse_method.get_all_recourse_instructions(poi)
        # Start the paths from the POI
        counterfactuals = []
        for instructions in all_instructions:
            counterfactual = self.adapter.interpret_instructions(poi, instructions)
            counterfactuals.append(counterfactual)
        paths = []
        # Finish the paths by iterating one path at a time
        for direction_index, counterfactual in enumerate(counterfactuals):
            rest_of_path = self.iterate_recourse_path(counterfactual, direction_index, max_iterations - 1)
            path = pd.concat([poi.to_frame().T, rest_of_path]).reset_index(drop=True)
            paths.append(path)
        return paths

    def iterate_recourse_path(self, poi: pd.Series, direction_index: int, max_iterations: int) -> pd.DataFrame:
        """Generates a recourse path for the model recourse direction given by direction_index."""
        path = [poi.to_frame().T]
        for i in range(max_iterations):
            if poi.isnull().any():
                raise RuntimeError(f"The iterated point has NaN values after {i} iterations. The point is:\n{poi}")
            if self.certainty_cutoff and self.check_model_certainty(self.adapter.transform_series(poi)) > self.certainty_cutoff:
                break
            instructions = self.recourse_method.get_kth_recourse_instructions(poi, direction_index)
            poi = self.adapter.interpret_instructions(poi, instructions)
            path.append(poi.to_frame().T)
        return pd.concat(path).reset_index(drop=True)

    def check_model_certainty(self, poi: recourse_adapter.EmbeddedSeries) -> float:
        return self.check_certainty(poi)
