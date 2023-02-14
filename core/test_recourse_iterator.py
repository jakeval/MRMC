import unittest
from unittest import mock
import pandas as pd
import numpy as np
from core import recourse_iterator


class TestRecourseIterator(unittest.TestCase):
    @mock.patch(
        "core.recourse_iterator.RecourseMethod",
        autospec=True,
    )
    @mock.patch(
        "core.recourse_iterator.recourse_adapter.RecourseAdapter",
        autospec=True,
    )
    @mock.patch("core.recourse_iterator.model_interface.Model", autospec=True)
    def test_iterate_recourse_path_cross_boundary(
        self,
        mock_model,
        mock_adapter,
        mock_recourse_method,
    ):
        mock_certainty = 0.6
        mock_poi = pd.Series([0, 1])
        mock_model.predict_pos_proba_series.side_effect = [0.2, 0.7]
        mock_adapter.interpret_instructions.return_value = mock_poi
        iterator = recourse_iterator.RecourseIterator(
            mock_recourse_method,
            mock_adapter,
            mock_certainty,
            mock_model,
        )
        points_in_path = iterator.iterate_recourse_path(mock_poi, 0, 10)
        # When the iterator crosses the decision boundary (0.7 > 0.6) in two
        # steps, then there should only be two steps in the path.
        self.assertEqual(2, len(points_in_path))

    @mock.patch(
        "core.recourse_iterator.RecourseMethod",
        autospec=True,
    )
    @mock.patch(
        "core.recourse_iterator.recourse_adapter.RecourseAdapter",
        autospec=True,
    )
    @mock.patch("core.recourse_iterator.model_interface.Model", autospec=True)
    def test_iterate_recourse_path_max_iterations(
        self,
        mock_model,
        mock_adapter,
        mock_recourse_method,
    ):
        mock_certainty = 0.6
        mock_poi = pd.Series([0, 1])
        mock_model.predict_pos_proba_series.return_value = 0
        mock_adapter.interpret_instructions.return_value = mock_poi
        iterator = recourse_iterator.RecourseIterator(
            mock_recourse_method,
            mock_adapter,
            mock_certainty,
            mock_model,
        )
        max_iterations = 3
        points_in_path = iterator.iterate_recourse_path(
            mock_poi, 0, max_iterations
        )
        # When the iterator doesn't cross the decision boundary, it should give
        # up after max_iterations. Max iterations counts iteration steps, so
        # max_iterations = 1 + path length
        self.assertEqual(max_iterations + 1, len(points_in_path))

    @mock.patch(
        "core.recourse_iterator.RecourseMethod",
        autospec=True,
    )
    @mock.patch(
        "core.recourse_iterator.recourse_adapter.RecourseAdapter",
        autospec=True,
    )
    @mock.patch("core.recourse_iterator.model_interface.Model", autospec=True)
    def test_iterate_recourse_path_has_null(
        self,
        mock_model,
        mock_adapter,
        mock_recourse_method,
    ):
        mock_certainty = 0.6
        mock_poi = pd.Series([0, np.nan])
        iterator = recourse_iterator.RecourseIterator(
            mock_recourse_method,
            mock_adapter,
            mock_certainty,
            mock_model,
        )
        with self.assertRaises(RuntimeError):
            iterator.iterate_recourse_path(mock_poi, 0, 10)

    @mock.patch(
        "core.recourse_iterator.RecourseMethod",
        autospec=True,
    )
    @mock.patch(
        "core.recourse_iterator.recourse_adapter.RecourseAdapter",
        autospec=True,
    )
    def test_iterate_k_recourse_paths(
        self,
        mock_adapter,
        mock_recourse_method,
    ):
        # one-dimensional data. recourse moves 0 -> 1
        mock_poi = pd.Series([0])
        mock_recourse_instructions = pd.Series([1])
        mock_next_point = pd.Series([1])

        # the mock recourse instructions move 0 -> 1
        mock_recourse_method.get_all_recourse_instructions.return_value = [
            mock_recourse_instructions
        ]
        mock_adapter.interpret_instructions.return_value = mock_next_point

        # RecourseIterator is mocked by passing in a mock self argument
        mock_self = mock.Mock(
            spec=["recourse_method", "adapter", "iterate_recourse_path"]
        )
        mock_self.recourse_method = mock_recourse_method
        mock_self.adapter = mock_adapter

        # continuing the path just terminates the path immediately
        mock_self.iterate_recourse_path.return_value = (
            mock_next_point.to_frame().T
        )

        paths = recourse_iterator.RecourseIterator.iterate_k_recourse_paths(
            mock_self, mock_poi, 10
        )

        # we expect there to be one path with points [[0], [1]]
        expected_path = pd.concat(
            [mock_poi.to_frame().T, mock_next_point.to_frame().T]
        )

        self.assertEqual(1, len(paths))
        np.testing.assert_equal(expected_path.to_numpy(), paths[0].to_numpy())

    @mock.patch(
        "core.recourse_iterator.RecourseMethod",
        autospec=True,
    )
    @mock.patch(
        "core.recourse_iterator.recourse_adapter.RecourseAdapter",
        autospec=True,
    )
    def test_iterate_k_recourse_paths_no_recourse(
        self,
        mock_adapter,
        mock_recourse_method,
    ):
        # one-dimensional data. recourse moves 0 -> 1
        mock_poi = pd.Series([0])
        mock_recourse_instructions = None  # No recourse is available

        mock_recourse_method.get_all_recourse_instructions.return_value = [
            mock_recourse_instructions
        ]

        # RecourseIterator is mocked by passing in a mock self argument
        mock_self = mock.Mock(
            spec=["recourse_method", "adapter", "iterate_recourse_path"]
        )
        mock_self.recourse_method = mock_recourse_method
        mock_self.adapter = mock_adapter

        paths = recourse_iterator.RecourseIterator.iterate_k_recourse_paths(
            mock_self, mock_poi, 10
        )

        # we expect there to be one stub path with point [[0]]
        expected_path = mock_poi.to_frame().T

        self.assertEqual(1, len(paths))
        np.testing.assert_equal(expected_path.to_numpy(), paths[0].to_numpy())
