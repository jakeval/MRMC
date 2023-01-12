import unittest
from unittest import mock

import numpy as np

from experiments import utils


class TestUtils(unittest.TestCase):
    @mock.patch("numpy.random.default_rng", autospec=True)
    @mock.patch("experiments.utils.model_selection", autospec=True)
    def test_create_run_configs(self, mock_parameter_grid, mock_default_rng):
        parameter_ranges = {"param1": [1, 2], "param2": ["a", "b"]}
        num_runs = 2
        random_seeds = np.array([0, 1])
        mock_default_rng().integers.return_value = random_seeds

        config_grid = [
            {"param1": 1, "param2": "a"},
            {"param1": 1, "param2": "b"},
            {"param1": 2, "param2": "a"},
            {"param1": 2, "param2": "b"},
        ]

        mock_parameter_grid.ParameterGrid.return_value = config_grid

        expected_configs = [
            {
                "param1": 1,
                "param2": "a",
                "batch_id": 0,
                "run_id": 0,
                "run_seed": 0,
            },
            {
                "param1": 1,
                "param2": "a",
                "batch_id": 0,
                "run_id": 1,
                "run_seed": 1,
            },
            {
                "param1": 1,
                "param2": "b",
                "batch_id": 1,
                "run_id": 2,
                "run_seed": 0,
            },
            {
                "param1": 1,
                "param2": "b",
                "batch_id": 1,
                "run_id": 3,
                "run_seed": 1,
            },
            {
                "param1": 2,
                "param2": "a",
                "batch_id": 2,
                "run_id": 4,
                "run_seed": 0,
            },
            {
                "param1": 2,
                "param2": "a",
                "batch_id": 2,
                "run_id": 5,
                "run_seed": 1,
            },
            {
                "param1": 2,
                "param2": "b",
                "batch_id": 3,
                "run_id": 6,
                "run_seed": 0,
            },
            {
                "param1": 2,
                "param2": "b",
                "batch_id": 3,
                "run_id": 7,
                "run_seed": 1,
            },
        ]

        run_configs = utils.create_run_configs(parameter_ranges, num_runs)

        for run_config, expected_config in zip(run_configs, expected_configs):
            self.assertEqual(run_config, expected_config)

    @mock.patch("numpy.random.default_rng", autospec=True)
    @mock.patch("experiments.utils.model_selection", autospec=True)
    def test_create_run_configs_seed(
        self, mock_parameter_grid, mock_default_rng
    ):
        parameter_ranges = {"param1": [1, 2], "param2": ["a", "b"]}
        num_runs = 2
        random_seed = 19301
        random_seeds = np.array([0, 1])
        mock_default_rng().integers.return_value = random_seeds

        config_grid = [
            {"param1": 1, "param2": "a"},
            {"param1": 1, "param2": "b"},
            {"param1": 2, "param2": "a"},
            {"param1": 2, "param2": "b"},
        ]

        mock_parameter_grid.ParameterGrid.return_value = config_grid

        expected_configs = [
            {
                "param1": 1,
                "param2": "a",
                "batch_id": 0,
                "run_id": 0,
                "run_seed": 0,
            },
            {
                "param1": 1,
                "param2": "a",
                "batch_id": 0,
                "run_id": 1,
                "run_seed": 1,
            },
            {
                "param1": 1,
                "param2": "b",
                "batch_id": 1,
                "run_id": 2,
                "run_seed": 0,
            },
            {
                "param1": 1,
                "param2": "b",
                "batch_id": 1,
                "run_id": 3,
                "run_seed": 1,
            },
            {
                "param1": 2,
                "param2": "a",
                "batch_id": 2,
                "run_id": 4,
                "run_seed": 0,
            },
            {
                "param1": 2,
                "param2": "a",
                "batch_id": 2,
                "run_id": 5,
                "run_seed": 1,
            },
            {
                "param1": 2,
                "param2": "b",
                "batch_id": 3,
                "run_id": 6,
                "run_seed": 0,
            },
            {
                "param1": 2,
                "param2": "b",
                "batch_id": 3,
                "run_id": 7,
                "run_seed": 1,
            },
        ]

        run_configs = utils.create_run_configs(
            parameter_ranges, num_runs, random_seed=random_seed
        )

        mock_default_rng.assert_called_with(random_seed)

        for run_config, expected_config in zip(run_configs, expected_configs):
            self.assertEqual(run_config, expected_config)
