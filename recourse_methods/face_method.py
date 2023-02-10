from typing import Sequence, Any, Optional, Tuple
import pandas as pd
import numpy as np
import numba
from scipy import sparse
from data import recourse_adapter
from recourse_methods import base_type
from models import model_interface


@numba.jit(nopython=True)
def _get_edge_weight(distance, weight_bias):
    return -np.log(distance) + weight_bias


class FACE(base_type.RecourseMethod):
    def __init__(
        self,
        dataset: pd.DataFrame,
        adapter: recourse_adapter.RecourseAdapter,
        model: model_interface.Model,
        k_directions: int,
        distance_threshold: float = 0.75,
        confidence_threshold: Optional[float] = None,
        graph_filepath: Optional[str] = None,
        counterfactual_mode: bool = True,
        weight_bias: float = 0,
    ):
        self.dataset = dataset
        self.adapter = adapter
        self.model = model
        self.confidence_threshold = confidence_threshold
        if graph_filepath:
            self.graph = sparse.load_npz(graph_filepath)
        else:
            self.graph = None
        self.candidate_indices = None
        self.distance_threshold = distance_threshold
        self.k_directions = k_directions
        self.counterfactual_mode = counterfactual_mode
        self.weight_bias = weight_bias

    def generate_graph(
        self,
        distance_threshold: float = 0.75,
        filepath_to_save_to: Optional[str] = None,
    ):
        """Generates and saves an epsilon-graph."""
        data = self.adapter.transform(
            self.dataset.drop(columns=self.adapter.label_column)
        )
        data.to_numpy()
        graph_weights = FACE._get_e_graph_weights(
            data.to_numpy(), distance_threshold
        )
        sparse_graph_weights = sparse.csr_array(graph_weights)
        if filepath_to_save_to is not None:
            sparse.save_npz(filepath_to_save_to, sparse_graph_weights)
        self.graph = sparse_graph_weights

    @staticmethod
    @numba.jit(nopython=True)
    def _get_e_graph_weights(
        embedded_data: np.ndarray, epsilon: float, weight_bias: float = 0
    ) -> np.ndarray:
        n_samples = embedded_data.shape[0]
        kernel = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i):
                pairwise_distance = np.linalg.norm(
                    embedded_data[i] - embedded_data[j]
                )
                if (pairwise_distance <= epsilon) and pairwise_distance > 0:
                    kernel[i, j] = _get_edge_weight(
                        pairwise_distance, weight_bias
                    )
                    kernel[j, i] = kernel[i, j]
        return kernel

    def fit(self):
        """Fits FACE to a dataset.

        It does this by finding the indices of each point to use as a candidate
        when performing graph search. These candidates are the potential
        counterfactuals returned by FACE."""
        candidate_mask = (
            self.dataset[self.adapter.label_column]
            == self.adapter.positive_label
        ).to_numpy()
        if self.confidence_threshold is not None:
            pos_proba = self.model.predict_pos_proba(self.dataset)
            candidate_mask = candidate_mask & (
                pos_proba >= self.confidence_threshold
            )
        self.candidate_indices = np.arange(self.dataset.shape[0])[
            candidate_mask
        ]
        return self

    def generate_paths(
        self, poi: recourse_adapter.EmbeddedSeries, num_paths: int, debug=False
    ) -> Sequence[recourse_adapter.EmbeddedDataFrame]:
        distances, predecessors = self.dijkstra_search(poi, self.graph)
        # return distances, predecessors
        target_indices = FACE.get_k_best_candidate_indices(
            distances, self.candidate_indices, num_paths
        )
        if debug:
            return distances, predecessors, target_indices
        return self._get_paths_from_indices(target_indices, poi, predecessors)

    def dijkstra_search(
        self, poi: recourse_adapter.EmbeddedSeries, graph: sparse.csr_array
    ):
        # temporarily add the POI to the graph so we can use it in graph search
        graph, new_index = self.append_new_point(poi, graph)
        distances, predecessors = sparse.csgraph.dijkstra(
            graph, indices=new_index, return_predecessors=True
        )
        # remove the temporarily added point from the results
        distances = np.hstack(
            [distances[:new_index], distances[new_index + 1 :]]
        )
        predecessors = np.hstack(
            [predecessors[:new_index], predecessors[new_index + 1 :]]
        )
        predecessors = np.where(predecessors == new_index, -1, predecessors)
        return distances, predecessors

    @staticmethod
    def get_k_best_candidate_indices(
        distances: np.ndarray,
        candidate_indices: np.ndarray,
        k_candidates: int,
    ) -> np.ndarray:
        """Attempts to return the indices of the k best candidate
        counterfactuals.

        "Best" means it minimizes the edge weights along the shortest path
        between it and the POI.

        Because the inputs and outputs of this function are numpy arrays, the
        indices it returns don't correspond to pandas indices of the source
        data (if any exist).

        If fewer than k viable candidates exist, it returns only as many as
        possible. If no candidates exist, it returns an empty list.

        If two candidates have the same distance, the candidate with lower
        index is preferred.

        Returns:
            A list of at most k indices.
        """
        sorted_candidate_indices = candidate_indices[
            np.argsort(distances[candidate_indices], kind="stable")
        ]
        candidate_indices = []
        for target_index in sorted_candidate_indices:
            if len(candidate_indices) == k_candidates:
                break
            if np.isinf(distances[target_index]):  # there is no path to here
                continue
            candidate_indices.append(target_index)
        return np.array(candidate_indices)

    def _get_paths_from_indices(
        self,
        target_indices: Sequence[int],
        poi: recourse_adapter.EmbeddedSeries,
        predecessors: Sequence[int],
    ) -> Sequence[recourse_adapter.EmbeddedDataFrame]:
        """Constructs the recourse paths given the counterfactuals and a
        predecessors array.

        Predecessors array: if point i preceeds j in a path, then
        predecessors[j] = i."""
        paths = []
        data = self.adapter.transform(
            self.dataset.drop(columns=self.adapter.label_column)
        ).to_numpy()
        for target_index in target_indices:
            path = []
            point_index = target_index
            while point_index != -1:
                path = [data[point_index]] + path
                point_index = predecessors[point_index]
            path = [poi.to_numpy()] + path
            paths.append(pd.DataFrame(columns=poi.index, data=np.vstack(path)))
        return paths

    def append_new_point(
        self, poi: recourse_adapter.EmbeddedSeries, graph: sparse.csr_array
    ) -> Tuple[sparse.csr_array, int]:
        """Creates a graph identical to its input but with the addition of the
        POI.

        It calculates the weighted distances between the POI and the other
        points in the graph and adds these distances to the graph by
        concatenating to the distance matrix.

        Returns:
            A new graph including the POI and the index at which the POI
            appears in the graph."""
        poi = poi.to_numpy()
        data = self.adapter.transform(
            self.dataset.drop(columns=self.adapter.label_column)
        ).to_numpy()
        distances = np.linalg.norm(data - poi, axis=1)

        # A mask for values we don't want to calculate weights on
        exclude_mask = (distances == 0) | (distances > self.distance_threshold)

        weights = distances.copy()

        # excluded values are 0
        weights[exclude_mask] = 0

        # all other values get the appropriate weight
        weights[~exclude_mask] = _get_edge_weight.pyfunc(
            distances[~exclude_mask], self.weight_bias
        )

        row_update = sparse.csr_array(weights)
        col_update = sparse.csr_array(np.hstack([weights, [0]])[None, :].T)
        graph = sparse.vstack([graph, row_update])
        graph = sparse.hstack([graph, col_update])
        return graph, graph.shape[0] - 1

    def _get_k_recourse_directions(
        self, poi: recourse_adapter.EmbeddedSeries, k_directions: int
    ) -> recourse_adapter.EmbeddedDataFrame:
        """Generates different recourse directions for the poi for each of the
        k_directions.

        Args:
            poi: The Point of Interest (POI) to find recourse directions for.

        Returns:
            A DataFrame containing recourse directions for the POI."""
        paths = self.generate_paths(poi, k_directions)
        if len(paths) == 0:
            return pd.DataFrame(columns=poi.index)
        recourse_points = []  # These can be counterfactuals or steps in a path
        for path in paths:
            if self.counterfactual_mode:
                recourse_points.append(path.iloc[-1].to_numpy())
            else:
                recourse_points.append(path.iloc[1].to_numpy())
        recourse_points = np.vstack(recourse_points)
        directions = recourse_points - poi.to_numpy()
        return pd.DataFrame(data=directions, columns=poi.index)

    def get_all_recourse_directions(
        self, poi: recourse_adapter.EmbeddedSeries
    ) -> recourse_adapter.EmbeddedDataFrame:
        """Generates different recourse directions for the poi for each of the
        k_directions.

        Args:
            poi: The Point of Interest (POI) to find recourse directions for.

        Returns:
            A DataFrame containing recourse directions for the POI."""
        return self._get_k_recourse_directions(poi, self.k_directions)

    def get_all_recourse_instructions(
        self, poi: pd.Series
    ) -> Sequence[Optional[Any]]:
        """Generates different recourse instructions for the poi for each of
        the k_directions.

        Whereas recourse directions are vectors in embedded space,
        instructions are human-readable guides for how to follow those
        directions in the original data space.

        Args:
            poi: The Point of Interest (POI) to find recourse instructions for.

        Returns:
            A Sequence of k recourse instructions for the POI. Elements may be
            None if there is no recourse available."""
        poi = self.adapter.transform_series(poi)

        # this may be an empty dataframe
        recourse_directions = self.get_all_recourse_directions(poi)
        instructions = []
        for i in range(recourse_directions.shape[0]):
            instructions.append(
                self.adapter.directions_to_instructions(
                    recourse_directions.iloc[i]
                )
            )
        if len(instructions) < self.k_directions:
            instructions += [None] * (self.k_directions - len(instructions))
        return instructions

    def get_kth_recourse_instructions(
        self, poi: pd.Series, dir_index: int
    ) -> Optional[Any]:
        """Generates a single set of recourse instructions for the kth
        direction.

        Args:
            poi: The Point of Interest (POI) to get the kth recourse
            instruction for.
            dir_index: Which of the k sets of instructions to return. FACE
                ignores this argument.

        Returns:
            Instructions for the POI to achieve the recourse. Returns None
            if no recourse is possible."""
        poi = self.adapter.transform_series(poi)
        recourse_directions = self._get_k_recourse_directions(poi, 1)
        if len(recourse_directions) == 0:
            return None
        return self.adapter.directions_to_instructions(
            recourse_directions.iloc[0]
        )
