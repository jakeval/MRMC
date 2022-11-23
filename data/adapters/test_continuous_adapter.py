import unittest
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
        for label, expected_label in zip(
            embedded_dataset.label, expected_labels
        ):
            self.assertEqual(label, expected_label)

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
        for recovered_label, expected_label in zip(
            recovered_dataset.label, mock_dataset.label
        ):
            self.assertEqual(recovered_label, expected_label)

        # the original values are recovered
        max_abs_difference = (
            np.abs(mock_dataset[["a", "b"]] - recovered_dataset[["a", "b"]])
            .max()
            .max()
        )
        self.assertAlmostEqual(max_abs_difference, 0)

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

        expected_dataset = mock_dataset.drop(columns=["label", "a"])

        # no errors are thrown
        max_abs_difference = np.abs(
            expected_dataset.b - recovered_dataset.b
        ).max()
        self.assertAlmostEqual(max_abs_difference, 0)
