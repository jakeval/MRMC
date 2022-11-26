import unittest
from unittest import mock
import numpy as np
from core import utils


class TestUtils(unittest.TestCase):
    @mock.patch("core.utils.np.random.normal", return_value=np.array([1, 0]))
    def test_randomly_perturb_direction(self, mock_normal: mock.Mock):
        new_direction = utils.randomly_perturb_direction(np.array([0, 1]), 1)
        expected_direction = np.array([0.70711, 0.70711])  # from sqrt(0.5)
        np.testing.assert_almost_equal(
            new_direction, expected_direction, decimal=5
        )
        self.assertAlmostEqual(np.linalg.norm(new_direction), 1.0)

    @mock.patch("core.utils.np.random.Generator", autospec=True)
    def test_randomly_perturb_direction_random_state(
        self, mock_generator: mock.Mock
    ):
        mock_generator.normal.return_value = np.array([1, 0])
        new_direction = utils.randomly_perturb_direction(
            np.array([0, 1]), 1, random_generator=mock_generator
        )
        expected_direction = np.array([0.70711, 0.70711])  # from sqrt(0.5)
        np.testing.assert_almost_equal(
            new_direction, expected_direction, decimal=5
        )
        self.assertAlmostEqual(np.linalg.norm(new_direction), 1.0)

    @mock.patch("core.utils.np.random.normal", return_value=np.array([1, 0]))
    def test_randomly_perturb_direction_zero_ratio(
        self, mock_normal: mock.Mock
    ):
        new_direction = utils.randomly_perturb_direction(np.array([0, 1]), 0)
        expected_direction = np.array([0, 1])
        np.testing.assert_almost_equal(new_direction, expected_direction)
        self.assertAlmostEqual(np.linalg.norm(new_direction), 1.0)

    @mock.patch("core.utils.np.random.normal", return_value=np.array([1, 0]))
    def test_randomly_perturb_direction_zero_direction(
        self, mock_normal: mock.Mock
    ):
        new_direction = utils.randomly_perturb_direction(np.array([0, 0]), 0.5)
        expected_direction = np.array([0, 0])
        np.testing.assert_almost_equal(new_direction, expected_direction)

    @mock.patch("core.utils.np.random.normal", return_value=np.array([0, 0]))
    def test_randomly_perturb_direction_zero_noise(
        self, mock_normal: mock.Mock
    ):
        new_direction = utils.randomly_perturb_direction(np.array([0, 1]), 0.5)
        expected_direction = np.array([0, 1])
        np.testing.assert_almost_equal(new_direction, expected_direction)

    def test_constant_step_size(self):
        new_direction = utils.constant_step_size(np.array([1, 1]), 0.5)
        expected_direction = np.array([0.35355, 0.35355])  # from sqrt(1/8)
        np.testing.assert_almost_equal(
            new_direction, expected_direction, decimal=5
        )
        self.assertAlmostEqual(np.linalg.norm(new_direction), 0.5)

    def test_constant_step_size_zero_direction(self):
        new_direction = utils.constant_step_size(np.array([0, 0]), 0.5)
        expected_direction = np.array([0, 0])
        np.testing.assert_almost_equal(new_direction, expected_direction)

    def test_constant_step_size_zero_step_size(self):
        new_direction = utils.constant_step_size(np.array([1, 1]), 0)
        expected_direction = np.array([0, 0])
        np.testing.assert_almost_equal(new_direction, expected_direction)
