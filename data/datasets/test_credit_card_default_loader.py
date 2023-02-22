import unittest
from unittest import mock
import numpy as np
from data.datasets import credit_card_default_loader


class TestCreditCardDefaultLoader(unittest.TestCase):
    @mock.patch(
        "data.datasets.credit_card_default_loader.np.random.Generator",
        autospec=True,
    )
    def test_split_indices(self, mock_rng):
        num_indices = 5
        train_ratio = 0.4
        val_ratio = 0.3
        test_ratio = 0.3

        # mocking the Generator disables shuffling the dataset before splitting

        splits = (
            credit_card_default_loader.CreditCardDefaultLoader._split_indices(
                num_indices, train_ratio, val_ratio, test_ratio
            )
        )

        expected_splits = {
            "train": np.array([0, 1]),  # 0.4 * 5 = 2 elements
            "val": np.array(
                [2]  # 0.3 * 5 = 1.5 -> 1 element.
            ),  # The extra element is passed to the next split.
            "test": np.array(
                [8, 9]  # 0.3 * 5 = 1.5 -> 1 element.
            ),  # It has 2 elements because the 'val' split passed it an extra.
        }

        for key in splits.keys():
            np.testing.assert_equal(splits[key], expected_splits[key])
