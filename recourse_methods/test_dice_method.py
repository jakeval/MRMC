import unittest
from unittest import mock
import pandas as pd
import numpy as np
from recourse_methods import dice_method


class TestDICE(unittest.TestCase):
    @mock.patch("dice_ml.Data", autospec=True)
    @mock.patch("dice_ml.Dice", autospec=True)
    @mock.patch("models.model_interface.Model", autospec=True)
    @mock.patch("data.recourse_adapter.RecourseAdapter", autospec=True)
    def test_init(
        self,
        mock_adapter: mock.Mock,
        mock_model: mock.Mock,
        mock_dice: mock.Mock,
        mock_dice_data: mock.Mock,
    ):
        mock_dice_kwargs = {"mock_arg": "mock_val"}
        mock_counterfactual_kwargs = {"mock_arg": "mock_val"}
        mock_adapter.label_column = "mock_label"
        mock_confidence = 0.5

        dice = dice_method.DiCE(
            k_directions=3,
            adapter=mock_adapter,
            dataset=None,
            continuous_features=None,
            model=mock_model,
            desired_confidence=mock_confidence,
            dice_kwargs=mock_dice_kwargs,
            dice_counterfactual_kwargs=mock_counterfactual_kwargs,
        )

        expected_counterfactual_kwargs = mock_counterfactual_kwargs.copy()
        expected_counterfactual_kwargs["stopping_threshold"] = mock_confidence
        self.assertEqual(
            expected_counterfactual_kwargs, dice.dice_counterfactual_kwargs
        )

        expected_dice_args = {
            "data_interface": mock_dice_data(),
            "model_interface": mock_model.to_dice_model(),
        }
        expected_dice_args.update(mock_dice_kwargs)
        mock_dice.assert_called_with(**expected_dice_args)

    @mock.patch(
        "recourse_methods.dice_method.recourse_adapter.RecourseAdapter",
        autospec=True,
    )
    def test_counterfactuals_to_directions(self, mock_adapter):
        mock_adapter.transform.side_effect = lambda df: df * 10
        mock_adapter.transform_series.side_effect = lambda series: series * 10
        mock_self = mock.Mock(spec=dice_method.DiCE)
        mock_self.adapter = mock_adapter

        poi = pd.Series([0, 1], index=["col1", "col2"])
        counterfactuals = pd.DataFrame(
            {"col1": [5, -2], "col2": [3, 2]}, index=[10, 23]
        )

        # multiple the POI and counterfactuals by 10 (because of the transform)
        # then subtract the POI from the counterfactuals.
        expected_directions = pd.DataFrame(
            {
                "col1": [50, -20],
                "col2": [20, 10],
            }
        )

        directions = dice_method.DiCE._counterfactuals_to_directions(
            mock_self, poi, counterfactuals
        )

        np.testing.assert_equal(
            directions.to_numpy(), expected_directions.to_numpy()
        )

    def test_format_dice_counterfactual_args(self):
        poi = pd.Series([0, 1], index=["col1", "col2"])
        num_counterfactuals = 2
        mock_counterfactual_kwargs = {"mock_arg": "mock_val"}
        mock_seed = 0
        expected_args = {
            "query_instances": pd.DataFrame({"col1": [0], "col2": [1]}),
            "total_CFs": 2,
            "desired_class": 1,
            "verbose": False,
            "mock_arg": "mock_val",
            "random_seed": mock_seed,
        }
        args = dice_method.DiCE._format_dice_counterfactual_args(
            poi, num_counterfactuals, mock_counterfactual_kwargs, mock_seed
        )

        self.assertEqual(set(expected_args.keys()), set(args.keys()))
        for arg in sorted(expected_args.keys()):
            val = args[arg]
            expected_val = expected_args[arg]
            if arg == "query_instances":
                np.testing.assert_equal(
                    val.to_numpy(), expected_val.to_numpy()
                )
            else:
                self.assertEqual(val, expected_val)
