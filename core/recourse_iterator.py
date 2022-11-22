from typing import Sequence, Optional
import pandas as pd
from recourse_methods.base_type import RecourseMethod
from models import model_interface
from data import recourse_adapter


class RecourseIterator:
    """Class for iterating recourse."""

    def __init__(
        self,
        recourse_method: RecourseMethod,
        adapter: recourse_adapter.RecourseAdapter,
        certainty_cutoff: Optional[float] = None,
        model: Optional[model_interface.Model] = None,
    ):
        """Creates a new recourse iterator.

        Args:
            recourse_method: The recourse method to use.
            adapter: A RecourseAdapter object to transform the data between
                human-readable and embedded space.
            certainty_cutoff: If not None, stop iterating early if the model
                certainty reaches the cutoff.
            model: The model to use to check positive outcome certainty if
                certainty_cutoff is not None.
        """
        self.recourse_method = recourse_method
        self.certainty_cutoff = certainty_cutoff
        self.model = model
        self.adapter = adapter

    # TODO(@jakeval): Confidence check
    def iterate_k_recourse_paths(
        self, poi: pd.Series, max_iterations: int
    ) -> Sequence[pd.DataFrame]:
        """Generates one recourse path for each of the model recourse
        directions.

        Args:
            poi: The Point of Interest (POI) to generate recourse paths for.
            max_iterations: The maximum number of steps in each path.

        Returns:
            A sequence of DataFrames where each DataFrame is a path and each
            row of a given DataFrame is a single step in the path.
        """
        all_instructions = self.recourse_method.get_all_recourse_instructions(
            poi
        )
        # Start the paths from the POI
        counterfactuals = []
        for instructions in all_instructions:
            counterfactual = self.adapter.interpret_instructions(
                poi, instructions
            )
            counterfactuals.append(counterfactual)
        paths = []
        # Finish the paths by iterating one path at a time
        for direction_index, counterfactual in enumerate(counterfactuals):
            rest_of_path = self.iterate_recourse_path(
                counterfactual, direction_index, max_iterations - 1
            )
            path = pd.concat([poi.to_frame().T, rest_of_path]).reset_index(
                drop=True
            )
            paths.append(path)
        return paths

    # TODO(@jakeval): Confidence check
    def iterate_recourse_path(
        self, poi: pd.Series, direction_index: int, max_iterations: int
    ) -> pd.DataFrame:
        """Generates a recourse path for the model recourse direction given by
        dir_index.

        Args:
            poi: The Point of Interest (POI) to generate a recourse path for.
            direction_index: The index of the path to generate.
            max_iterations: The maximum number of steps in the path.
        Returns:
            A DataFrame where each row of the DataFrame is a step in the
            path.
        """
        path = [poi.to_frame().T]
        for i in range(max_iterations):
            if poi.isnull().any():
                raise RuntimeError(
                    (
                        f"The iterated point has NaN values after {i} "
                        f"iterations. The point is:\n{poi}"
                    )
                )
            if (
                self.certainty_cutoff
                and self.model.predict_pos_proba_series(poi)
                > self.certainty_cutoff
            ):
                break
            instructions = self.recourse_method.get_kth_recourse_instructions(
                poi, direction_index
            )
            poi = self.adapter.interpret_instructions(poi, instructions)
            path.append(poi.to_frame().T)
        return pd.concat(path).reset_index(drop=True)
