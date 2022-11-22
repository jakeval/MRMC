import unittest
from unittest import mock
import numpy as np
from core import utils


class TestUtils(unittest.TestCase):
    @mock.patch("core.utils.np.random.normal", return_value=np.array([1, 0]))
    def test_randomly_perturb_direction(self, mock_normal: mock.Mock):
        new_direction = utils.randomly_perturb_direction(np.array([0, 1]), 1)
        expected_direction = np.array([np.sqrt(0.5), np.sqrt(0.5)])
        for new_val, expected_val in zip(new_direction, expected_direction):
            self.assertAlmostEqual(new_val, expected_val)
        self.assertAlmostEqual(
            np.linalg.norm(new_direction), np.linalg.norm(np.array([0, 1]))
        )

    @mock.patch("core.utils.np.random.normal", return_value=np.array([1, 0]))
    def test_randomly_perturb_direction_zero_ratio(
        self, mock_normal: mock.Mock
    ):
        new_direction = utils.randomly_perturb_direction(np.array([0, 1]), 0)
        expected_direction = np.array([0, 1])
        for new_val, expected_val in zip(new_direction, expected_direction):
            self.assertAlmostEqual(new_val, expected_val)
        self.assertAlmostEqual(
            np.linalg.norm(new_direction), np.linalg.norm(np.array([0, 1]))
        )

    @mock.patch("core.utils.np.random.normal", return_value=np.array([1, 0]))
    def test_randomly_perturb_direction_zero_direction(
        self, mock_normal: mock.Mock
    ):
        new_direction = utils.randomly_perturb_direction(np.array([0, 0]), 0.5)
        expected_direction = np.array([0, 0])
        for new_val, expected_val in zip(new_direction, expected_direction):
            self.assertAlmostEqual(new_val, expected_val)

    @mock.patch("core.utils.np.random.normal", return_value=np.array([0, 0]))
    def test_randomly_perturb_direction_zero_noise(
        self, mock_normal: mock.Mock
    ):
        new_direction = utils.randomly_perturb_direction(np.array([0, 1]), 0.5)
        expected_direction = np.array([0, 1])
        for new_val, expected_val in zip(new_direction, expected_direction):
            self.assertAlmostEqual(new_val, expected_val)

    def test_constant_step_size(self):
        new_direction = utils.constant_step_size(np.array([1, 1]), 0.5)
        expected_direction = np.array([np.sqrt(1 / 8), np.sqrt(1 / 8)])
        for new_val, expected_val in zip(new_direction, expected_direction):
            self.assertAlmostEqual(new_val, expected_val)
        self.assertAlmostEqual(np.linalg.norm(new_direction), 0.5)

    def test_constant_step_size_zero_direction(self):
        new_direction = utils.constant_step_size(np.array([0, 0]), 0.5)
        expected_direction = np.array([0, 0])
        for new_val, expected_val in zip(new_direction, expected_direction):
            self.assertAlmostEqual(new_val, expected_val)

    def test_constant_step_size_zero_step_size(self):
        new_direction = utils.constant_step_size(np.array([1, 1]), 0)
        expected_direction = np.array([0, 0])
        for new_val, expected_val in zip(new_direction, expected_direction):
            self.assertAlmostEqual(new_val, expected_val)
