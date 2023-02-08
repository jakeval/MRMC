import unittest

import pandas as pd
import numpy as np
from experiments.mrmc_experiment import run_mrmc_experiment


class TestMRMCExperiment(unittest.TestCase):
    def test_merge_results(self):
        all_results = {
            "result_1": pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
            "result_2": pd.DataFrame({"a": [3]}),
        }
        new_results = {
            "result_1": pd.DataFrame({"a": [4, 5, 6], "b": [7, 8, 9]}),
            "result_2": pd.DataFrame({"a": [1]}),
        }

        expected_results = {
            "result_1": pd.DataFrame(
                {"a": [1, 2, 3, 4, 5, 6], "b": [4, 5, 6, 7, 8, 9]}
            ),
            "result_2": pd.DataFrame({"a": [3, 1]}),
        }

        merged_results = run_mrmc_experiment.merge_results(
            all_results, new_results
        )

        self.assertEqual(
            set(expected_results.keys()), set(merged_results.keys())
        )

        for key, merged_result in merged_results.items():
            expected_result = expected_results[key]
            np.testing.assert_equal(
                merged_result.to_numpy(), expected_result.to_numpy()
            )
