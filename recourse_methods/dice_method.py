from typing import Sequence, Optional, Mapping, Any
from recourse_methods import base_type
import pandas as pd
from data import data_preprocessor as dp
import dice_ml
from dice_ml import constants
from sklearn import pipeline


class ToNumpy:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def transform(self, X):
        return X.to_numpy()


class DiCE(base_type.RecourseMethod):
    """An abstract base class for recourse methods."""

    def __init__(self,
                 k_directions: int,
                 preprocessor: dp.Preprocessor,
                 dataset: pd.DataFrame,
                 continuous_features: Sequence[str],
                 model: Any,
                 model_backend: constants.BackEndTypes,
                 label_column: str = 'Y',
                 dice_kwargs: Optional[Mapping[str, Any]] = None,
                 dice_cfe_kwargs: Optional[Mapping[str, Any]] = None):
        """Constructs a new DiCE recourse method.
        
        Args:
            k_directions: The number of recourse directions to generate.
            preprocessor: The dataset preprocessor.
            dataset: The dataset to perform recourse over.
            continuous_features: A list of the dataset's continuous features.
            model: An ML model satisfying one of the DiCE model backends.
            model_backend: Model types recognized by DiCE (sklearn, pytorch, and tensorflow).
            label_column: The column containing binary classification outputs.
            dice_kwargs: Optional arguments to pass to DiCE on instantiation.
            dice_recourse_kwargs: Optional arguments to pass to DiCE on counterfactual explanation generation.
        """
        self.k_directions = k_directions
        self.preprocessor = preprocessor
        self.dice_cfe_kwargs = dice_cfe_kwargs
        self.label_column = label_column
        d = dice_ml.Data(
            dataframe=dataset,
            continuous_features=continuous_features,
            outcome_name=label_column
        )
        clf = pipeline.Pipeline(steps=[('preprocessor', preprocessor),
                                       ('tonumpy', ToNumpy()),
                                       ('classifier', model)])
        m = dice_ml.Model(model=clf, backend=model_backend)
        dice_args = {
            'data_interface': d,
            'model_interface': m,
        }
        if dice_kwargs:
            dice_args.update(dice_kwargs)
        self.dice = dice_ml.Dice(**dice_args)

    def get_all_recourse_directions(self, poi: dp.EmbeddedSeries) -> dp.EmbeddedDataFrame:
        """Generates different recourse directions for the poi for each of the k_directions."""
        poi = self.preprocessor.inverse_transform_series(poi)
        cfes = self._generate_counterfactuals(poi, self.k_directions)
        directions = self._counterfactuals_to_directions(poi, cfes)
        return directions

    def get_all_recourse_instructions(self, poi: pd.Series) -> Sequence[Any]:
        """Generates different recourse instructions for the poi for each of the k_directions."""
        cfes = self._generate_counterfactuals(poi, self.k_directions)
        directions = self._counterfactuals_to_directions(poi, cfes)
        instructions = []
        for i in range(len(cfes)):
            instruction = self.preprocessor.directions_to_instructions(directions.iloc[i])
            instructions.append(instruction)
        return instructions

    def get_kth_recourse_instructions(self, poi: pd.Series, dir_index: int) -> Any:
        """Generates a single set of recourse instructions for the kth direction."""
        cfes = self._generate_counterfactuals(poi, 1)
        directions = self._counterfactuals_to_directions(poi, cfes)
        return self.preprocessor.directions_to_instructions(directions.iloc[0])

    def _generate_counterfactuals(self, poi: pd.Series, num_cfes: int) -> pd.DataFrame:
        """Generates DiCE counterfactual examples for the requested POI."""
        cfe_args = {
            'query_instances': poi.to_frame().T,
            'total_CFs': num_cfes,
            'desired_class': 'opposite',
            'verbose': False
        }
        if self.dice_cfe_kwargs:
            cfe_args.update(self.dice_cfe_kwargs)
        return self.dice.generate_counterfactuals(**cfe_args).cf_examples_list[0].final_cfs_df.drop(self.label_column, axis=1)

    def _counterfactuals_to_directions(self, poi: pd.Series, cfes: pd.DataFrame) -> dp.EmbeddedDataFrame:
        """Converts a DataFrame of counterfactual points to a DataFrame of Embedded directions pointing from the POI to the CFEs."""
        poi = self.preprocessor.transform_series(poi)
        cfes = self.preprocessor.transform(cfes)
        dirs = cfes - poi
        return dirs
