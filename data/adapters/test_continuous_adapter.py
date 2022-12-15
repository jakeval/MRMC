import unittest
from unittest import mock
import pandas as pd
import numpy as np
from data.adapters import continuous_adapter


class TestStandardizingAdapter(unittest.TestCase):
    def test_transform(self):
        label_column = "label"
        positive_label = "positive"
        mock_dataset = pd.DataFrame(
            {
                "a": [0, 1, 2],
                "b": [-2, 4, 6],
                "label": ["positive", "negative", "negative"],
            }
        )
        expected_labels = [1, -1, -1]

        embedded_dataset = (
            continuous_adapter.StandardizingAdapter(
                label_column, positive_label=positive_label
            )
            .fit(mock_dataset)
            .transform(mock_dataset)
        )

        # mean should be 0
        for mean in embedded_dataset[["a", "b"]].mean():
            self.assertAlmostEqual(mean, 0)

        # standard deviation should be 1
        for std in np.std(embedded_dataset[["a", "b"]]):
            self.assertAlmostEqual(std, 1)

        # labels should be {-1,1} encoded
        np.testing.assert_equal(
            embedded_dataset.label.to_numpy(), expected_labels
        )

    def test_transform_missing_columns(self):
        label_column = "label"
        positive_label = "positive"
        mock_dataset = pd.DataFrame(
            {
                "a": [0, 1, 2],
                "b": [-2, 4, 6],
                "label": ["positive", "negative", "negative"],
            }
        )

        embedded_dataset = (
            continuous_adapter.StandardizingAdapter(
                label_column, positive_label=positive_label
            )
            .fit(mock_dataset)
            .transform(mock_dataset.drop(columns=["a", "label"]))
        )

        # no error is thrown for missing columns
        self.assertAlmostEqual(embedded_dataset.b.mean(), 0)
        self.assertAlmostEqual(np.std(embedded_dataset.b), 1)

    def test_inverse_transform(self):
        label_column = "label"
        positive_label = "positive"
        a_col = np.array([0, 1, 2])
        b_col = np.array([-2, 4, 6])
        mock_dataset = pd.DataFrame(
            {
                "a": a_col,
                "b": b_col,
                "label": ["positive", "negative", "negative"],
            }
        )
        mock_embedded_dataset = pd.DataFrame(
            {
                "a": (a_col - a_col.mean()) / a_col.std(),
                "b": (b_col - b_col.mean()) / b_col.std(),
                "label": [1, -1, -1],
            }
        )

        recovered_dataset = (
            continuous_adapter.StandardizingAdapter(
                label_column=label_column, positive_label=positive_label
            )
            .fit(mock_dataset)
            .inverse_transform(mock_embedded_dataset)
        )

        # original labels are recovered
        np.testing.assert_equal(
            recovered_dataset.label.to_numpy(), mock_dataset.label.to_numpy()
        )

        # the original values are recovered
        np.testing.assert_almost_equal(
            mock_dataset[["a", "b"]].to_numpy(),
            recovered_dataset[["a", "b"]].to_numpy(),
        )

    def test_inverse_transform_missing_columns(self):
        label_column = "label"
        positive_label = "positive"
        a_col = np.array([0, 1, 2])
        b_col = np.array([-2, 4, 6])
        mock_dataset = pd.DataFrame(
            {
                "a": a_col,
                "b": b_col,
                "label": ["positive", "negative", "negative"],
            }
        )
        mock_embedded_dataset = pd.DataFrame(
            {
                "b": (b_col - b_col.mean()) / b_col.std(),
            }
        )

        recovered_dataset = (
            continuous_adapter.StandardizingAdapter(
                label_column=label_column, positive_label=positive_label
            )
            .fit(mock_dataset)
            .inverse_transform(mock_embedded_dataset)
        )

        # no errors are thrown
        expected_dataset = mock_dataset.drop(columns=["label", "a"])

        np.testing.assert_almost_equal(
            expected_dataset.b.to_numpy(), recovered_dataset.b.to_numpy()
        )

    @mock.patch(
        "data.adapters.continuous_adapter.utils.randomly_perturb_direction",
        autospec=True,
    )
    def test_interpret_instructions(self, mock_perturb):
        mock_self = mock.Mock(spec=continuous_adapter.StandardizingAdapter)
        mock_self.perturb_ratio = None
        mock_self.rescale_ratio = None
        mock_self.transform_series.side_effect = lambda series: series
        mock_self.inverse_transform_series.side_effect = (
            lambda series: series + 1
        )
        poi = pd.Series([0, 1])
        instructions = pd.Series([1, -1])
        # We expect poi + instructions + 1 because of the inverse transform.
        expected_counterfactual_example = pd.Series([2, 1])
        counterfactual_example = (
            continuous_adapter.StandardizingAdapter.interpret_instructions(
                mock_self, poi=poi, instructions=instructions
            )
        )

        self.assertTrue(
            (expected_counterfactual_example == counterfactual_example).all()
        )
        mock_perturb.assert_not_called()

    @mock.patch(
        "data.adapters.continuous_adapter.utils.randomly_perturb_direction",
        autospec=True,
    )
    def test_interpret_instructions_perturb(self, mock_perturb):
        mock_self = mock.Mock(spec=continuous_adapter.StandardizingAdapter)
        mock_self.perturb_ratio = 0.5
        mock_self.rescale_ratio = None
        mock_self.transform_series.side_effect = lambda series: series
        mock_self.inverse_transform_series.side_effect = (
            lambda series: series + 1
        )
        poi = pd.Series([0, 1])
        instructions = pd.Series([1, -1])
        mock_perturb.return_value = pd.Series([1, 1])

        # perturb(poi) + instructions + 1 because of the inverse transform.
        expected_counterfactual_example = pd.Series([2, 3])

        counterfactual_example = (
            continuous_adapter.StandardizingAdapter.interpret_instructions(
                mock_self, poi=poi, instructions=instructions
            )
        )

        self.assertTrue(
            (expected_counterfactual_example == counterfactual_example).all()
        )
        mock_perturb.assert_called_with(instructions, 0.5)

    @mock.patch(
        "data.adapters.continuous_adapter.utils.randomly_perturb_direction",
        autospec=True,
    )
    def test_interpret_instructions_rescale(self, mock_perturb):
        mock_self = mock.Mock(spec=continuous_adapter.StandardizingAdapter)
        mock_self.perturb_ratio = None
        mock_self.rescale_ratio = 0.5
        mock_self.transform_series.side_effect = lambda series: series
        mock_self.inverse_transform_series.side_effect = (
            lambda series: series + 1
        )
        poi = pd.Series([0, 1])
        instructions = pd.Series([1, -1])

        # instructions/2 + 1 because of the rescale and inverse transform.
        expected_counterfactual_example = pd.Series([1.5, 1.5])

        counterfactual_example = (
            continuous_adapter.StandardizingAdapter.interpret_instructions(
                mock_self, poi=poi, instructions=instructions
            )
        )

        self.assertTrue(
            (expected_counterfactual_example == counterfactual_example).all()
        )
        mock_perturb.assert_not_called()
