import unittest
from unittest import mock
import numpy as np
import pandas as pd
from recourse_methods import mrmc_method


_SMALL_CUTOFF = 0.2
_LARGE_CUTOFF = 2
_DEGREE = 2
volcano_alpha_large_cutoff = mrmc_method.get_volcano_alpha(
    cutoff=_LARGE_CUTOFF, degree=_DEGREE
)
volcano_alpha_small_cutoff = mrmc_method.get_volcano_alpha(
    cutoff=_SMALL_CUTOFF, degree=_DEGREE
)


class TestMRMC(unittest.TestCase):
    def test_volcano_alpha_small_cutoff(self):
        distances = np.array([0, 1, 2, 3])
        weights = volcano_alpha_small_cutoff(distances)
        expected_weights = np.array(
            [
                1 / (_SMALL_CUTOFF**_DEGREE),
                1 / (1**_DEGREE),
                1 / (2**_DEGREE),
                1 / (3**_DEGREE),
            ]
        )
        np.testing.assert_almost_equal(weights, expected_weights)

    def test_volcano_alpha_large_cutoff(self):
        distances = np.array([0, 1, 2, 3])
        weights = volcano_alpha_large_cutoff(distances)
        expected_weights = np.array(
            [
                1 / (_LARGE_CUTOFF**_DEGREE),
                1 / (_LARGE_CUTOFF**_DEGREE),
                1 / (2**_DEGREE),
                1 / (3**_DEGREE),
            ]
        )
        np.testing.assert_almost_equal(weights, expected_weights)

    @mock.patch(
        "data.recourse_adapter.RecourseAdapter",
        autospec=True,
    )
    def test_process_data(self, mock_adapter: mock.Mock):
        mock_adapter.positive_label = 1
        mock_adapter.label_column = "label"
        mock_adapter.transform.side_effect = lambda df: df
        dataset = pd.DataFrame(
            {"col_1": [1, 2, 3], "col_2": [3, 5, 3], "label": [1, -1, 1]}
        )
        processed_data = mrmc_method.MRM.process_data(dataset, mock_adapter)
        # The label should be dropped
        self.assertNotIn("label", processed_data.columns)
        # There should be no negative examples remaining
        negative_label_indices = (dataset[dataset.label == -1]).index
        self.assertEqual(
            0, len(processed_data.index.intersection(negative_label_indices))
        )
        # The transform function should be called
        mock_adapter.transform.assert_called_once()

    def test_get_unnormalized_direction(self):
        mock_self = mock.Mock(spec=["data", "alpha"])
        poi = pd.Series([0, 0], index=["a", "b"])
        mock_self.data = pd.DataFrame({"a": [1, 1], "b": [-1, 1]})
        mock_self.alpha.side_effect = [pd.Series([0.75, 0.25])]
        expected_direction = pd.Series(
            [1, 0.75 * -1 + 0.25 * 1], index=["a", "b"]
        )
        direction = mrmc_method.MRM.get_unnormalized_direction(mock_self, poi)
        for val, expected_val in zip(direction, expected_direction):
            self.assertEqual(val, expected_val)

    def test_get_unnormalized_direction_nan_alpha(self):
        mock_self = mock.Mock(spec=["data", "alpha"])
        poi = pd.Series([0, 0], index=["a", "b"])
        mock_self.data = pd.DataFrame({"a": [1, 1], "b": [-1, 1]})
        mock_self.alpha.side_effect = [pd.Series([np.nan, 0.25])]
        with self.assertRaises(RuntimeError):
            mrmc_method.MRM.get_unnormalized_direction(mock_self, poi)
