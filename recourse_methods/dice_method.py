from typing import Sequence, Optional, Mapping, Any
from recourse_methods import base_type
import pandas as pd
from data import recourse_adapter
import dice_ml
from dice_ml import constants
from sklearn import pipeline


# TODO(@jakeval): This will be refactored with the model update.
class ToNumPy:
    """This is a temporary class used until model handling is refactored.

    It is used in an sklearn pipeline to convert pandas DataFrames to NumPy
    arrays.
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def transform(self, X):
        return X.to_numpy()


class DiCE(base_type.RecourseMethod):
    """An abstract base class for recourse methods."""

    def __init__(
        self,
        k_directions: int,
        adapter: recourse_adapter.RecourseAdapter,
        dataset: pd.DataFrame,
        continuous_features: Sequence[str],
        model: Any,
        model_backend: constants.BackEndTypes,
        label_column: str = "Y",
        dice_kwargs: Optional[Mapping[str, Any]] = None,
        dice_cfe_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        """Constructs a new DiCE recourse method.

        The RecourseAdapter translates between the original data format
        (potentially with categorical features) and a continuous embedded
        space. Recourse directions are generated in embedded space and
        interpreted as instructions in the original data format.

        The failure modes for this class are identical to those for the DiCE
        method. Namely, for some points and optimizer parameter settings, DiCE
        will return counterfactual examples which do not cross the decision
        boundary.

        Args:
            k_directions: The number of recourse directions to generate.
            adapter: The dataset adapter.
            dataset: The dataset to perform recourse over.
            continuous_features: A list of the dataset's continuous features.
            model: An ML model satisfying one of the DiCE model backends.
            model_backend: Model types recognized by DiCE (sklearn, pytorch,
                and tensorflow).
            label_column: The column containing binary classification outputs.
            dice_kwargs: Optional arguments to pass to DiCE on instantiation.
            dice_recourse_kwargs: Optional arguments to pass to DiCE on
                counterfactual explanation generation.
        """
        self.k_directions = k_directions
        self.adapter = adapter
        self.dice_cfe_kwargs = dice_cfe_kwargs
        self.label_column = label_column
        d = dice_ml.Data(
            dataframe=dataset,
            continuous_features=continuous_features,
            outcome_name=label_column,
        )
        clf = pipeline.Pipeline(
            steps=[
                ("adapter", adapter),
                ("tonumpy", ToNumPy()),
                ("classifier", model),
            ]
        )
        m = dice_ml.Model(model=clf, backend=model_backend)
        dice_args = {
            "data_interface": d,
            "model_interface": m,
        }
        if dice_kwargs:
            dice_args.update(dice_kwargs)
        self.dice = dice_ml.Dice(**dice_args)

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
        cfes = self._generate_counterfactuals(poi, self.k_directions)
        directions = self._counterfactuals_to_directions(poi, cfes)
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
        cfes = self._generate_counterfactuals(poi, self.k_directions)
        directions = self._counterfactuals_to_directions(poi, cfes)
        instructions = []
        for i in range(len(cfes)):
            instruction = self.adapter.directions_to_instructions(
                directions.iloc[i]
            )
            instructions.append(instruction)
        return instructions

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
        cfes = self._generate_counterfactuals(poi, 1)
        directions = self._counterfactuals_to_directions(poi, cfes)
        return self.adapter.directions_to_instructions(directions.iloc[0])

    def _generate_counterfactuals(
        self, poi: pd.Series, num_cfes: int
    ) -> pd.DataFrame:
        """Generates DiCE counterfactual examples (CFEs) for the requested POI.

        Args:
            poi: The Point of Interest to generate counterfactual examples for.

        Returns:
            A DataFrame of counterfactual examples.
        """
        cfe_args = {
            "query_instances": poi.to_frame().T,
            "total_CFs": num_cfes,
            "desired_class": "opposite",
            "verbose": False,
        }
        if self.dice_cfe_kwargs:
            cfe_args.update(self.dice_cfe_kwargs)
        return (
            self.dice.generate_counterfactuals(**cfe_args)
            .cf_examples_list[0]
            .final_cfs_df.drop(self.label_column, axis=1)
        )

    def _counterfactuals_to_directions(
        self, poi: pd.Series, cfes: pd.DataFrame
    ) -> recourse_adapter.EmbeddedDataFrame:
        """Converts a DataFrame of counterfactual points to a DataFrame of
        Embedded directions pointing from the POI to the CFEs.

        Args:
            poi: The Point of Interest (POI) the counterfactual examples were
                generated for.
            cfes: The counterfactual examples (CFEs) generated for the POI.

        Returns:
            A DataFrame of recourse directions in embedded space.
        """
        poi = self.adapter.transform_series(poi)
        cfes = self.adapter.transform(cfes)
        dirs = cfes - poi
        return dirs
