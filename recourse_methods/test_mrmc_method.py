import unittest
import numpy as np
from recourse_methods import mrmc_method


SMALL_CUTOFF = 0.2
LARGE_CUTOFF = 2
DEGREE = 2
volcano_alpha_large_cutoff = mrmc_method.get_volcano_alpha(
    cutoff=LARGE_CUTOFF, degree=DEGREE
)
volcano_alpha_small_cutoff = mrmc_method.get_volcano_alpha(
    cutoff=SMALL_CUTOFF, degree=DEGREE
)


class TestUtils(unittest.TestCase):
    def test_volcano_alpha_small_cutoff(self):
        distances = np.array([0, 1, 2, 3])
        weights = volcano_alpha_small_cutoff(distances)
        expected_weights = np.array(
            [
                1 / (SMALL_CUTOFF**DEGREE),
                1 / (1**DEGREE),
                1 / (2**DEGREE),
                1 / (3**DEGREE),
            ]
        )

        for val, expected_val in zip(weights, expected_weights):
            self.assertAlmostEqual(val, expected_val)

    def test_volcano_alpha_large_cutoff(self):
        distances = np.array([0, 1, 2, 3])
        weights = volcano_alpha_large_cutoff(distances)
        expected_weights = np.array(
            [
                1 / (LARGE_CUTOFF**DEGREE),
                1 / (LARGE_CUTOFF**DEGREE),
                1 / (2**DEGREE),
                1 / (3**DEGREE),
            ]
        )

        for val, expected_val in zip(weights, expected_weights):
            self.assertAlmostEqual(val, expected_val)
