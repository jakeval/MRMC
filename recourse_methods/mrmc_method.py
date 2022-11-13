from __future__ import annotations
from typing import Any, Protocol, Sequence
from dataclasses import dataclass
import numpy as np
from core import utils
from models import model_interface
import pandas as pd
from data import recourse_adapter
from recourse_methods.base_type import RecourseMethod
from sklearn.cluster import KMeans


def get_volcano_alpha(cutoff=0.5, degree=2) -> AlphaFunction:
    """Returns a volcano-shaped alpha function.

    Args:
        cutoff: The smallest distance after which the alpha output is constant.
        degree: The degree of the exponential.

    Returns:
        An alpha function initialized by cutoff and degree.
    """

    def volcano_alpha(dist: np.ndarray) -> np.ndarray:
        nonlocal cutoff, degree
        return 1 / np.where(dist <= cutoff, cutoff, dist) ** degree

    return volcano_alpha


def normalize_rescaler(
    mrm: MRM, direction: recourse_adapter.EmbeddedSeries
) -> recourse_adapter.EmbeddedSeries:
    """Normalizes an MRM direction based on dataset size.

    Args:
        mrm: An MRM instance containing necessary dataset information.
        direction: The direction to normalize.

    Returns:
        A normalized copy of direction.
    """
    return direction / len(mrm.data)


def get_constant_step_size_rescaler(step_size: float = 1) -> RecourseRescaler:
    """Returns a rescaling function which scales directions to a constant size.

    Args:
        step_size: The step size to rescale to.

    Returns:
        A constant step size RecourseRescaler function.
    """
    return lambda _, direction: utils.constant_step_size(
        direction, step_size=step_size
    )


class RecourseRescaler(Protocol):
    """A data type for recourse rescaling used by MRM."""

    def rescale(
        mrm: MRM, direction: recourse_adapter.EmbeddedSeries
    ) -> recourse_adapter.EmbeddedSeries:
        """Rescales a recourse direction generated by MRM.

        Args:
            mrm: The MRM instance which generated the direction to rescale.
            direction: The recourse direction to rescale.

        Returns:
            A rescaled recourse direction.
        """


class AlphaFunction(Protocol):
    """A data type for alpha functions used by MRM."""

    def alpha(dist: np.ndarray) -> np.ndarray:
        """Given a list containing distances, returns a list of weights for
        each distance.

        Args:
            dist: The list of distances.

        Returns:
            A list of weights of the same size as dist.
        """


class MRM:
    """Monotone Recourse Measures generates recourse directions."""

    def __init__(
        self,
        dataset: pd.DataFrame,
        adapter: recourse_adapter.RecourseAdapter,
        alpha: AlphaFunction = get_volcano_alpha(),
        rescale_direction: RecourseRescaler = normalize_rescaler,
    ):
        """Creates an MRM instance.

        Args:
            dataset: The dataset to provide recourse over.
            adapter: A RecourseAdapter object to transform the data between
                human-readable and embedded space.
            alpha: The alpha function to use during recourse generation.
            rescale_direction: A function for rescaling the recourse.
        """
        self.data = MRM.process_data(dataset, adapter)
        self.adapter = adapter
        self.alpha = alpha
        self.rescale_direction = rescale_direction

    @staticmethod
    def process_data(
        dataset: pd.DataFrame,
        adapter: recourse_adapter.RecourseAdapter,
    ) -> recourse_adapter.EmbeddedDataFrame:
        """Processes the dataset for MRM.
        It filters out negative outcomes and transforms the data to a numeric
        embedded space.

        Args:
            dataset: The dataset to process.
            adapter: The recourse adapter which transforms data to a numeric
                     embedded space.

        Returns:
            A processed dataset.
        """
        positive_mask = dataset[adapter.label_name] == adapter.positive_label
        positive_dataset = dataset[positive_mask]
        positive_embedded_dataset = adapter.transform(
            positive_dataset.drop(adapter.label_name, axis=1)
        )
        if len(positive_embedded_dataset) == 0:
            raise ValueError(
                "Dataset is empty after excluding negative outcome examples."
            )
        return positive_embedded_dataset

    def filter_data(
        self, confidence_threshold: float, model: model_interface.Model
    ) -> MRM:
        """Filters the recourse dataset to include only high-confidence points.

        Args:
            confidence_threshold: All examples that do not achieve at least
                                  this model confidence are removed.
            model: The model to use for generating model confidence.

        Returns:
            Itself. The filtering is done mutably, so the returned version is
            not a copy.
        """
        p = model.predict_proba(self.data)
        self.data = self.data[p >= confidence_threshold]
        return self

    def get_unnormalized_direction(
        self, poi: recourse_adapter.EmbeddedSeries
    ) -> recourse_adapter.EmbeddedSeries:
        """Returns an unnormalized recourse direction.

        This is the core computation of MRM.

        Args:
            poi: The Point of Interest (POI) to generate an MRM direction for.

        Returns:
            An MRM direction in embedded space.
        """
        diff = (self.data - poi).to_numpy()
        dist = np.sqrt(np.power(diff, 2).sum(axis=1))
        alpha_val = self.alpha(dist)
        direction = diff.T @ alpha_val
        return recourse_adapter.EmbeddedSeries(index=poi.index, data=direction)

    def get_recourse_direction(
        self, poi: recourse_adapter.EmbeddedSeries
    ) -> recourse_adapter.EmbeddedSeries:
        """Returns the (maybe normalized) recourse direction in embedded space.
        If self.rescale_direction is not None, rescales the unnormalized
        direction. Typically rescaling is used to normalize the direction by
        dataset size.

        Args:
            poi: The Point of Interest (POI) to generate an MRM direction for.

        Returns:
            A possibly normalized MRM direction in embedded space.
        """
        direction = self.get_unnormalized_direction(poi)
        if self.rescale_direction:
            direction = self.rescale_direction(self, direction)
        return direction

    def get_recourse_instructions(self, poi: pd.Series) -> Any:
        """Returns recourse instructions for a given POI.

        Args:
            poi: The Point of Interest (poi) to generate instructions for.

        Returns:
            Instructions for how to translate the POI based on an MRM
            direction.
        """
        direction = self.get_recourse_direction(
            self.adapter.transform_series(poi)
        )
        return self.adapter.directions_to_instructions(direction)


@dataclass
class Clusters:
    """Class for kmeans cluster-related data.

    Attributes:
        cluster_assignments: A DataFrame containing columns 'datapoint_index'
            and 'datapoint_cluster'.
        cluster_centers: A numpy array containing the cluster centers.
    """

    cluster_assignments: pd.DataFrame
    cluster_centers: np.ndarray


class MRMC(RecourseMethod):
    """A class for Monotonic Recourse Measures (MRM) with clustering.

    MRMC clusters the data and initializes a separate MRM instance for
    each cluster.
    """

    def __init__(
        self,
        k_directions: int,
        adapter: recourse_adapter.RecourseAdapter,
        dataset: pd.DataFrame,
        alpha: AlphaFunction = get_volcano_alpha(),
        rescale_direction: RecourseRescaler = normalize_rescaler,
        clusters: Clusters = None,
    ):
        """Creates a new MRMC instance.

        Args:
            k_directions: The number of clusters (and recourse directions) to
                generate.
            adapter: A RecourseAdapter object to transform the data between
                human-readable and embedded space.
            dataset: The dataset to perform recourse over.
            alpha: The alpha function each MRM should use.
            rescale_direction: The rescaling function each MRM should use.
            clusters: The cluster data to use. If None, performs k-means
                clustering.
        """
        data = MRM.process_data(dataset, adapter)
        self.k_directions = k_directions
        if not clusters:
            clusters = self.cluster_data(data, self.k_directions)
        self.clusters = clusters
        self.validate_cluster_assignments(
            clusters.cluster_assignments, self.k_directions
        )

        mrms = []
        for cluster_index in range(k_directions):
            indices = clusters.cluster_assignments[
                clusters.cluster_assignments["datapoint_cluster"]
                == cluster_index
            ]["datapoint_index"]
            data_cluster = data.loc[indices]
            dataset_cluster = dataset.loc[data_cluster.index]
            mrm = MRM(
                dataset=dataset_cluster,
                adapter=adapter,
                alpha=alpha,
                rescale_direction=rescale_direction,
            )
            mrms.append(mrm)
        self.mrms: Sequence[MRM] = mrms

    def filter_data(
        self, confidence_threshold: float, model: model_interface.Model
    ) -> MRMC:
        """Filters the recourse dataset to include only high-confidence points.

        Args:
            confidence_threshold: All examples that do not achieve at least
                                  this model confidence are removed.
            model: The model to use for generating model confidence.

        Returns:
            Itself. The filtering is done mutably, so the returned version is
            not a copy.
        """
        for mrm in self.mrms:
            mrm.filter_data(confidence_threshold, model)
            return self

    def cluster_data(
        self, data: recourse_adapter.EmbeddedDataFrame, k_directions: int
    ) -> Clusters:
        """Clusters the data using k-means clustering.

        Args:
            data: The data to cluster in embedded continuous space.
            k_directions: The number of clusters to generate.
        Returns:
            The data Clusters.
        """
        km = KMeans(n_clusters=k_directions)
        cluster_assignments = km.fit_predict(data.to_numpy())
        cluster_assignments = np.vstack(
            [data.index.to_numpy(), cluster_assignments]
        ).T
        cluster_assignments_df = pd.DataFrame(
            columns=["datapoint_index", "datapoint_cluster"],
            data=cluster_assignments,
        )
        cluster_centers = km.cluster_centers_
        return Clusters(
            cluster_assignments=cluster_assignments_df,
            cluster_centers=cluster_centers,
        )

    def validate_cluster_assignments(
        self, cluster_assignments: pd.DataFrame, k_directions: int
    ) -> bool:
        """Raises an error if the cluster_assignments is valid.

        Invalid cluster_assignments have more or fewer clusters than the
        requested k_directions.

        Args:
            cluster_assignments: A dataframe of cluster assignments as
                described in the Clusters class.
            k_directions: The number of clusters expected.

        Raises:
            RuntimeError if the cluster assignment is invalid.

        Returns:
            True if the cluster assignment is valid.
        """
        cluster_sizes = cluster_assignments.groupby(
            "datapoint_cluster"
        ).count()
        if set(cluster_sizes.index) != set(range(k_directions)):
            raise RuntimeError(
                (
                    f"Data was assigned to clusters {cluster_sizes.index}"
                    f", but expected clusters {range(k_directions)}"
                )
            )
        return True

    def get_all_recourse_directions(
        self, poi: recourse_adapter.EmbeddedSeries
    ) -> recourse_adapter.EmbeddedDataFrame:
        """Generates different recourse directions for the poi for each of the
        k_directions.

        Args:
            poi: The Point of Interest (POI) to generate recourse directions
                for.

        Returns:
            A dataframe where each row is a different recourse direction.
        """
        directions = []
        for mrm in self.mrms:
            directions.append(mrm.get_recourse_direction(poi).to_frame().T)
        return pd.concat(directions).reset_index(drop=True)

    def get_all_recourse_instructions(self, poi: pd.Series) -> Sequence[Any]:
        """Generates different recourse instructions for the poi for each of
        the k_directions.

        Args:
            poi: The Point of Interest (poi) to generate recourse instructions
                for.

        Returns:
            A list where each element is a different recourse instruction.
        """
        instructions = []
        for mrm in self.mrms:
            instructions.append(mrm.get_recourse_instructions(poi))
        return instructions

    def get_kth_recourse_instructions(
        self, poi: pd.Series, direction_index: int
    ) -> Any:
        """Generates a single set of recourse instructions for the kth
        direction."""
        return self.mrms[direction_index].get_recourse_instructions(poi)


def check_directions(
    poi: recourse_adapter.EmbeddedSeries,
    directions: recourse_adapter.EmbeddedDataFrame,
    cluster_centers: np.ndarray,
) -> np.ndarray:
    """Convenience function for verifying the directions generated by MRMC.

    Returns the dot product between the normalized directions and the
    normalized difference vector between the POI and cluster centers. The
    smaller the dot products, the more likely the directions are correct. This
    is because MRM directions typically point towards the cluster centers.

    The dirs and cluster_centers should be aligned such that the ith direction
    corresponds to the ith cluster center.

    Args:
        poi: The Point of Interest (POI) used to generate recourse.
        dirs: The recourse directions generated.
        cluster_centers: An array where each row is a cluster center.

    Returns:
        An array where each element is the dot product between a recourse
        direction and its corresponding cluster center.
    """
    expected_directions = cluster_centers - poi.to_numpy()
    expected_directions = (
        expected_directions
        / np.linalg.norm(expected_directions, axis=1)[:, None]
    )
    actual_directions = directions.to_numpy()
    actual_directions = (
        actual_directions / np.linalg.norm(actual_directions, axis=1)[:, None]
    )
    return np.diag(expected_directions @ actual_directions.T)
