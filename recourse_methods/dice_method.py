from typing import Sequence, Optional, Mapping, Any
from recourse_methods import base_type
import pandas as pd
from data import recourse_adapter
import dice_ml
from models import model_interface


class DiCE(base_type.RecourseMethod):
    """An abstract base class for recourse methods.

    Recourse directions are generated in embedded space and
    interpreted as instructions in the original data format.

    The failure modes for this class are identical to those for the DiCE
    method. Namely, for some points and optimizer parameter settings, DiCE
    will return counterfactual examples which do not cross the decision
    boundary.
    """

    # TODO(@jakeval): Unit test to ensure dice initialization uses kwargs
    def __init__(
        self,
        k_directions: int,
        adapter: recourse_adapter.RecourseAdapter,
        dataset: pd.DataFrame,
        continuous_features: Sequence[str],
        model: model_interface.Model,
        desired_confidence: Optional[float] = None,
        dice_kwargs: Optional[Mapping[str, Any]] = None,
        dice_counterfactual_kwargs: Optional[Mapping[str, Any]] = None,
        random_seed: Optional[int] = None,
    ):
        """Constructs a new DiCE recourse method.

        The RecourseAdapter translates between the original data format
        (potentially with categorical features) and a continuous embedded
        space.

        Args:
            k_directions: The number of recourse directions to generate.
            adapter: A RecourseAdapter object to transform the data between
                human-readable and embedded space.
            dataset: The dataset to perform recourse over.
            continuous_features: A list of the dataset's continuous features.
            model: An ML model satisfying one of the DiCE model backends.
            dice_kwargs: Optional arguments to pass to DiCE on instantiation.
            dice_counterfactual_kwargs: Optional arguments to pass to DiCE on
                counterfactual explanation generation.
            random_seed: A random seed used to initialize DICE and generate
                deterministic recourse.
        """
        self.k_directions = k_directions
        self.adapter = adapter
        dice_counterfactual_kwargs = dice_counterfactual_kwargs or {}
        if desired_confidence:
            dice_counterfactual_kwargs[
                "stopping_threshold"
            ] = desired_confidence
        self.dice_counterfactual_kwargs = dice_counterfactual_kwargs
        d = dice_ml.Data(
            dataframe=dataset,
            continuous_features=continuous_features,
            outcome_name=adapter.label_column,
        )
        dice_args = {
            "data_interface": d,
            "model_interface": model.to_dice_model(),
        }
        if dice_kwargs:
            dice_args.update(dice_kwargs)
        self.dice = dice_ml.Dice(**dice_args)
        self.random_seed = random_seed

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
        poi = self.adapter.inverse_transform_series(poi)
        counterfactuals = self._generate_counterfactuals(
            poi, self.k_directions
        )
        directions = self._counterfactuals_to_directions(poi, counterfactuals)
        return directions

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
        counterfactuals = self._generate_counterfactuals(
            poi, self.k_directions
        )
        directions = self._counterfactuals_to_directions(poi, counterfactuals)
        instructions = []
        for i in range(len(counterfactuals)):
            instruction = self.adapter.directions_to_instructions(
                directions.iloc[i]
            )
            instructions.append(instruction)
        return instructions

    def get_kth_recourse_instructions(
        self, poi: pd.Series, direction_index: int
    ) -> Any:
        """Generates a single set of recourse instructions for the kth
        direction.

        Args:
            poi: The Point of Interest (POI) to get the kth recourse
                instruction for.
            dir_index: The index of the recourse direction to generate
                instructions for. This argument is ignored for DiCE.

        Returns:
            Instructions for the POI to achieve the recourse.
        """
        counterfactuals = self._generate_counterfactuals(poi, 1)
        directions = self._counterfactuals_to_directions(poi, counterfactuals)
        return self.adapter.directions_to_instructions(directions.iloc[0])

    def _generate_counterfactuals(
        self, poi: pd.Series, num_counterfactuals: int
    ) -> pd.DataFrame:
        """Generates DiCE counterfactual examples (CFEs) for the requested POI.

        Args:
            poi: The Point of Interest to generate counterfactual examples for.
            num_cfes: The number of counterfactual examples (CFEs) to generate.

        Returns:
            A DataFrame of counterfactual examples.
        """
        counterfactual_args = {
            "query_instances": poi.to_frame().T,
            "total_CFs": num_counterfactuals,
            "desired_class": self.adapter.positive_label,
            "verbose": False,
        }
        if self.random_seed:
            counterfactual_args.update({"random_seed": self.random_seed})
        if self.dice_counterfactual_kwargs:
            counterfactual_args.update(self.dice_counterfactual_kwargs)
        return (
            self.dice.generate_counterfactuals(**counterfactual_args)
            .cf_examples_list[0]
            .final_cfs_df.drop(self.adapter.label_column, axis=1)
        )

    #  TODO(@jakeval): Unit test this
    def _counterfactuals_to_directions(
        self, poi: pd.Series, counterfactuals: pd.DataFrame
    ) -> recourse_adapter.EmbeddedDataFrame:
        """Converts a DataFrame of counterfactual points to a DataFrame of
        Embedded directions pointing from the POI to the counterfactual
        examples.

        Args:
            poi: The Point of Interest (POI) the counterfactual examples were
                generated for.
            counterfactuals: The counterfactual examples (CFEs) generated for
                the POI.

        Returns:
            A DataFrame of recourse directions in embedded space.
        """
        poi = self.adapter.transform_series(poi)
        counterfactuals = self.adapter.transform(counterfactuals)
        directions = counterfactuals - poi
        return directions
