import unittest
from unittest import mock
import numpy as np
import pandas as pd
from recourse_methods import mrmc_method


_SMALL_CUTOFF = 0.2
_LARGE_CUTOFF = 2
_DEGREE = 2


class TestMRMC(unittest.TestCase):
    def test_volcano_alpha_small_cutoff(self):
        distances = np.array([0, 1, 2, 3])
        weights = mrmc_method.get_volcano_alpha(
            cutoff=_SMALL_CUTOFF, degree=_DEGREE
        )(distances)
        expected_weights = np.array(
            [
                1 / (_SMALL_CUTOFF**_DEGREE),
                1 / (1**_DEGREE),
                1 / (2**_DEGREE),
                1 / (3**_DEGREE),
            ]
        )
        np.testing.assert_almost_equal(weights, expected_weights)

    def test_volcano_alpha_large_cutoff(self):
        distances = np.array([0, 1, 2, 3])
        weights = mrmc_method.get_volcano_alpha(
            cutoff=_LARGE_CUTOFF, degree=_DEGREE
        )(distances)
        expected_weights = np.array(
            [
                1 / (_LARGE_CUTOFF**_DEGREE),
                1 / (_LARGE_CUTOFF**_DEGREE),
                1 / (2**_DEGREE),
                1 / (3**_DEGREE),
            ]
        )
        np.testing.assert_almost_equal(weights, expected_weights)

    @mock.patch(
        "data.recourse_adapter.RecourseAdapter",
        autospec=True,
    )
    def test_process_data(self, mock_adapter: mock.Mock):
        mock_adapter.positive_label = 1
        mock_adapter.label_column = "label"
        mock_adapter.transform.side_effect = lambda df: df
        dataset = pd.DataFrame(
            {"col_1": [1, 2, 3], "col_2": [3, 5, 3], "label": [1, -1, 1]}
        )
        processed_data = mrmc_method.MRM._process_data(dataset, mock_adapter)
        # The label should be dropped
        self.assertNotIn("label", processed_data.columns)
        # There should be no negative examples remaining
        negative_label_indices = (dataset[dataset.label == -1]).index
        self.assertEqual(
            0, len(processed_data.index.intersection(negative_label_indices))
        )
        # The transform function should be called
        mock_adapter.transform.assert_called_once()

    @mock.patch(
        "data.recourse_adapter.RecourseAdapter",
        autospec=True,
    )
    @mock.patch("models.model_interface.Model", autospec=True)
    def test_process_data_confidence_threshold(
        self, mock_model: mock.Mock, mock_adapter: mock.Mock
    ):
        mock_adapter.positive_label = 1
        mock_adapter.label_column = "label"
        mock_threshold = 0.7
        mock_adapter.transform.side_effect = lambda df: df
        dataset = pd.DataFrame(
            {"col_1": [1, 2, 3], "col_2": [3, 5, 3], "label": [1, -1, 1]}
        )
        mock_confidences = pd.Series([0.8, 0.9, 0.6])
        mock_model.predict_pos_proba.side_effect = [mock_confidences]
        processed_data = mrmc_method.MRM._process_data(
            dataset,
            mock_adapter,
            confidence_threshold=mock_threshold,
            model=mock_model,
        )
        # The label should be dropped
        self.assertNotIn("label", processed_data.columns)
        # There should be no negative examples remaining
        negative_label_indices = (dataset[dataset.label == -1]).index
        self.assertEqual(
            0, len(processed_data.index.intersection(negative_label_indices))
        )
        # There should be no low-confidence points
        low_confidence_indices = [2]
        self.assertEqual(
            0, len(processed_data.index.intersection(low_confidence_indices))
        )
        # The transform function should be called
        mock_adapter.transform.assert_called_once()

    def test_get_unnormalized_direction(self):
        mock_self = mock.Mock(spec=["data", "alpha"])
        poi = pd.Series([0, 0], index=["a", "b"])
        mock_self.data = pd.DataFrame({"a": [1, 1], "b": [-1, 1]})
        mock_self.alpha.side_effect = [pd.Series([0.75, 0.25])]
        expected_direction = pd.Series(
            [1, 0.75 * -1 + 0.25 * 1], index=["a", "b"]
        )
        direction = mrmc_method.MRM.get_unnormalized_direction(mock_self, poi)
        np.testing.assert_equal(
            direction.to_numpy(), expected_direction.to_numpy()
        )

    def test_get_unnormalized_direction_nan_alpha(self):
        mock_self = mock.Mock(spec=["data", "alpha"])
        poi = pd.Series([0, 0], index=["a", "b"])
        mock_self.data = pd.DataFrame({"a": [1, 1], "b": [-1, 1]})
        mock_self.alpha.side_effect = [pd.Series([np.nan, 0.25])]
        with self.assertRaises(RuntimeError):
            mrmc_method.MRM.get_unnormalized_direction(mock_self, poi)

    @mock.patch("recourse_methods.mrmc_method.KMeans", autospec=True)
    def test_cluster_data(self, mock_kmeans):
        mock_data = pd.DataFrame({"a": range(7), "b": range(7)})
        n_clusters = 3
        expected_cluster_assignments = np.array([0, 0, 0, 1, 1, 2, 2])
        expected_cluster_centers = "mock centers"
        mock_kmeans().fit_predict.return_value = expected_cluster_assignments
        mock_kmeans().cluster_centers_ = expected_cluster_centers
        seed = 199318

        clusters = mrmc_method.MRMC._cluster_data(
            mock_data, n_clusters, random_seed=seed
        )

        expected_cluster_assignment_df = pd.DataFrame(
            {
                "datapoint_index": range(7),
                "datapoint_cluster": expected_cluster_assignments,
            }
        )

        self.assertTrue(
            (clusters.cluster_assignments == expected_cluster_assignment_df)
            .all()
            .all()
        )
        self.assertEqual(clusters.cluster_centers, expected_cluster_centers)
        mock_kmeans.assert_called_with(
            n_clusters=n_clusters, random_state=seed
        )

    @mock.patch(
        "recourse_methods.mrmc_method.MRM._process_data", autospec=True
    )
    @mock.patch(
        "recourse_methods.mrmc_method.MRMC._cluster_data", autospec=True
    )
    @mock.patch(
        "recourse_methods.mrmc_method.MRMC._validate_cluster_assignments",
        autospec=True,
    )
    def test_mrmc_init(
        self, mock_validate, mock_clustering, mock_process_data
    ):
        mock_dataset = pd.DataFrame(
            {"a": [1, 2, 3, 4, 5, 6, 7], "b": [7, 6, 5, 4, 3, 2, 1]},
            index=range(7),
        )

        # processing is a no-op
        mock_process_data.return_value = mock_dataset

        # assign clusters
        mock_cluster_assignments = pd.DataFrame(
            {
                "datapoint_index": range(7),
                "datapoint_cluster": [0, 0, 0, 1, 1, 2, 2],
            }
        )
        mock_clusters = mrmc_method.Clusters(
            cluster_assignments=mock_cluster_assignments, cluster_centers=None
        )
        mock_clustering.return_value = mock_clusters
        seed = 193572

        mrmc = mrmc_method.MRMC(
            k_directions=3,
            adapter=None,
            dataset=mock_dataset,
            confidence_threshold=None,
            model=None,
            random_seed=seed,
        )

        # Check that _process_data was called with correct arguments.
        mock_process_data.assert_called_with(
            mock_dataset,
            None,
            confidence_threshold=None,
            model=None,
        )

        # Check that _cluster_data was called with filtered data.
        mock_clustering.assert_called_with(mock_dataset, 3, random_seed=seed)

        # Check that each MRM instance has the expected data indices.
        expected_mrm_data_indices = [[0, 1, 2], [3, 4], [5, 6]]
        for direction_index, mrm in enumerate(mrmc.mrms):
            self.assertTrue(
                (
                    mrm.data.index
                    == expected_mrm_data_indices[direction_index]
                ).all()
            )

    @mock.patch(
        "recourse_methods.mrmc_method.MRM._process_data", autospec=True
    )
    @mock.patch(
        "recourse_methods.mrmc_method.MRMC._cluster_data", autospec=True
    )
    @mock.patch(
        "recourse_methods.mrmc_method.MRMC._validate_cluster_assignments",
        autospec=True,
    )
    def test_mrmc_init_confidence_threshold(
        self, mock_validate, mock_clustering, mock_process_data
    ):
        mock_threshold = 0.6

        mock_dataset = pd.DataFrame(
            {"a": [1, 2, 3, 4, 5, 6, 7], "b": [7, 6, 5, 4, 3, 2, 1]},
            index=range(7),
        )

        # Let data processing filter out indices 0 and 3
        mock_processed_data = mock_dataset.iloc[[1, 2, 4, 5, 6]]
        mock_process_data.return_value = mock_processed_data

        # Assign clusters on remaining indices
        mock_cluster_assignments = pd.DataFrame(
            {
                "datapoint_index": [1, 2, 4, 5, 6],
                "datapoint_cluster": [0, 0, 1, 2, 2],
            }
        )
        mock_clusters = mrmc_method.Clusters(
            cluster_assignments=mock_cluster_assignments, cluster_centers=None
        )
        mock_clustering.return_value = mock_clusters

        seed = 18349572

        mrmc = mrmc_method.MRMC(
            k_directions=3,
            adapter=None,
            dataset=mock_dataset,
            confidence_threshold=mock_threshold,
            model=None,
            random_seed=seed,
        )

        # Check that _process_data was called with correct arguments.
        mock_process_data.assert_called_with(
            mock_dataset,
            None,
            confidence_threshold=mock_threshold,
            model=None,
        )

        # Check that _cluster_data was called with filtered data.
        mock_clustering.assert_called_with(
            mock_processed_data, 3, random_seed=seed
        )

        # Check that each MRM instance has the expected data indices.
        expected_mrm_data_indices = [[1, 2], [4], [5, 6]]
        for direction_index, mrm in enumerate(mrmc.mrms):
            self.assertTrue(
                (
                    mrm.data.index
                    == expected_mrm_data_indices[direction_index]
                ).all()
            )
