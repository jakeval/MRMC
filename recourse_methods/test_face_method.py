import unittest
from unittest import mock
import numpy as np
import pandas as pd
from scipy import sparse
from recourse_methods import face_method


class TestFACE(unittest.TestCase):
    def test_get_e_graph_weights(self):
        # Dataset is a square in 2D space
        dataset = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).astype(np.float64)
        w = -np.log(1)  # The weight of a unit distance

        # The edges should be made across unit distances, but not across the
        # hypotenuse.
        expected_edge_weights = np.array(
            [[0, w, w, 0], [w, 0, 0, w], [w, 0, 0, w], [0, w, w, 0]]
        ).astype(np.float64)

        edge_weights = face_method.Face._get_e_graph_weights(
            embedded_data=dataset, epsilon=1
        )
        np.testing.assert_almost_equal(expected_edge_weights, edge_weights)

    @mock.patch(
        "recourse_methods.face_method.recourse_adapter.RecourseAdapter"
    )
    @mock.patch(
        "recourse_methods.face_method.model_interface.Model", autospec=True
    )
    def test_fit(self, mock_model, mock_adapter):

        confidence_threshold = 0.6

        # The only high-confidence true positive is the last example
        dataset = pd.DataFrame({"label": [0, 1, 1]})
        mock_model.predict_pos_proba.return_value = np.array(
            [
                0,
                0,
                confidence_threshold,
            ]
        )
        mock_adapter.label_column = "label"
        mock_adapter.positive_label = 1

        # Prepare the mock face_method
        mock_self = mock.Mock(
            spec=[
                "dataset",
                "adapter",
                "model",
                "confidence_threshold",
                "candidate_indices",
            ]
        )
        mock_self.dataset = dataset
        mock_self.adapter = mock_adapter
        mock_self.model = mock_model
        mock_self.confidence_threshold = confidence_threshold

        # The only candidate index should be 2 (true positive, high confidence)
        expected_candidate_indices = np.array([2])

        face_method.Face.fit(mock_self)

        np.testing.assert_equal(
            mock_self.candidate_indices, expected_candidate_indices
        )

    def test_get_k_best_candidate_indices(self):
        # test when everything works

        # There are four points, but only the last three are candidates.
        distances = np.array([1, 2, 2, 1])
        candidate_indices = np.array([1, 2, 3])
        k_candidates = 2

        # Index 3 is chosen first because it is a candidate and has minimal
        # weight.
        # Index 1 is chosen second because it is a candidate and its index is
        # lower than the alternative it ties with.
        expected_candidate_indices = np.array([3, 1])

        candidate_indices = face_method.Face.get_k_best_candidate_indices(
            distances, candidate_indices, k_candidates
        )

        np.testing.assert_equal(candidate_indices, expected_candidate_indices)

    def test_get_k_best_candidate_indices_failure(self):
        # test when everything works

        # There are four points, but only the last three are candidates.
        distances = np.array([1, np.inf, np.inf, 1])
        candidate_indices = np.array([1, 2, 3])
        k_candidates = 2

        # Index 3 is chosen first because it is a candidate and has minimal
        # weight.
        # No other indices can be chosen -- the other candidates have infinite
        # weight.
        expected_candidate_indices = np.array([3])

        candidate_indices = face_method.Face.get_k_best_candidate_indices(
            distances, candidate_indices, k_candidates
        )

        np.testing.assert_equal(candidate_indices, expected_candidate_indices)

    @mock.patch(
        "recourse_methods.face_method.recourse_adapter.RecourseAdapter",
        autospec=True,
    )
    def test_get_paths_from_indices(self, mock_adapter):
        dataset = pd.DataFrame({"a": [1, 2, 3], "label": [0, 1, 1]})
        poi = pd.Series([0], index=["a"])
        target_indices = [2]
        predecessors = [
            -9999,  # there is no path to this point
            -1,  # this point is connected to the POI
            1,  # the path to here runs through point 1
        ]

        # The adapter is the identity function
        mock_adapter.label_column = "label"
        mock_adapter.transform.side_effect = lambda x: x

        mock_self = mock.Mock(spec=["adapter", "dataset"])
        mock_self.adapter = mock_adapter
        mock_self.dataset = dataset

        # The expected path goes through the POI, the point at index 1, and the
        # point at index 2.
        expected_paths = [pd.DataFrame({"a": [0, 2, 3]})]

        paths = face_method.Face._get_paths_from_indices(
            mock_self, target_indices, poi, predecessors
        )

        self.assertEqual(len(paths), len(expected_paths))

        pd.testing.assert_frame_equal(paths[0], expected_paths[0])

    @mock.patch(
        "recourse_methods.face_method.recourse_adapter.RecourseAdapter",
        autospec=True,
    )
    @mock.patch("recourse_methods.face_method._get_edge_weight")
    def test_append_new_point(self, mock_get_edge_weight, mock_adapter):
        distance_threshold = 1
        dataset = pd.DataFrame({"a": [0, 1, 2], "label": [0, 1, 1]})

        # The graph to append a new point to
        graph = sparse.csr_array(
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]]  # distance matrix
        )

        # The new point to append
        poi = pd.Series([0], index=["a"])

        # The edge weight function is 1/x. This is the identity for 1 and
        # inf for 0, which is good because append_new_point should check for
        # and avoid applying weights to 0-distance edges.
        # It references pyfunc because it is a numba function.
        mock_get_edge_weight.pyfunc.side_effect = (
            lambda distances, bias: 1 / distances
        )

        # The adapter is the identity function
        mock_adapter.label_column = "label"
        mock_adapter.transform.side_effect = lambda x: x

        mock_self = mock.Mock(
            spec=["adapter", "dataset", "distance_threshold", "weight_bias"]
        )
        mock_self.adapter = mock_adapter
        mock_self.dataset = dataset
        mock_self.distance_threshold = distance_threshold
        mock_self.weight_bias = 0

        expected_new_graph = np.array(
            [
                [0, 1, 0, 0],  # The distance is 0
                [1, 0, 1, 1],  # The distance is 1
                [0, 1, 0, 0],  # The distance is 2, which is set to 0
                [0, 1, 0, 0],  # The distance is 0
            ]
        )

        expected_new_point_index = 3

        new_graph, new_point_index = face_method.Face.append_new_point(
            mock_self, poi, graph
        )

        self.assertEqual(new_point_index, expected_new_point_index)
        np.testing.assert_equal(new_graph.toarray(), expected_new_graph)

    def test_get_k_recourse_directions_no_recourse(self):
        poi = pd.Series([0], index=["a"])
        mock_self = mock.Mock(spec=["generate_paths"])
        mock_self.generate_paths.return_value = []  # no recourse returned
        directions = face_method.Face._get_k_recourse_directions(
            mock_self, poi, 2
        )

        # The directions dataframe should be empty with columns ["a"]
        np.testing.assert_equal(directions.columns, np.array(["a"]))
        self.assertEqual(0, len(directions))

    @mock.patch(
        "recourse_methods.face_method.recourse_adapter.RecourseAdapter",
        autospec=True,
    )
    def test_get_all_recourse_instructions_nones(self, mock_adapter):
        poi = pd.Series([0], index=["a"])

        # adapter is the identity function
        mock_adapter.directions_to_instructions.side_effect = lambda x: x

        mock_self = mock.Mock(
            spec=["adapter", "get_all_recourse_directions", "k_directions"]
        )
        mock_self.adapter = mock_adapter
        mock_self.k_directions = 2  # two directions are requested
        mock_self.get_all_recourse_directions.return_value = pd.DataFrame(
            {"a": [1]}  # but only one direction is returned
        )

        expected_instructions = [pd.Series([1], index=["a"], name=0), None]

        instructions = face_method.Face.get_all_recourse_instructions(
            mock_self, poi
        )

        # First instruction is just the direction returned
        pd.testing.assert_series_equal(
            instructions[0], expected_instructions[0]
        )

        # Second instruction is None
        self.assertIsNone(instructions[1])

    @mock.patch(
        "recourse_methods.face_method.recourse_adapter.RecourseAdapter",
        autospec=True,
    )
    def test_get_kth_recourse_instructions(self, mock_adapter):
        poi = pd.Series([0], index=["a"])

        mock_self = mock.Mock(spec=["adapter", "_get_k_recourse_directions"])
        mock_self.adapter = mock_adapter
        mock_self._get_k_recourse_directions.return_value = pd.DataFrame(
            {"a": []}  # no recourse is available
        )
        instructions = face_method.Face.get_kth_recourse_instructions(
            mock_self, poi, 0
        )

        # None is returned because there is no recourse
        self.assertIsNone(instructions)
