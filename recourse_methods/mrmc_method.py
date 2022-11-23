from __future__ import annotations
from typing import Any, Protocol, Sequence, Optional
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
    if len(mrm.data) == 0:
        raise RuntimeError("Can't normalize direction against 0-lengh data.")
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
    """Monotone Recourse Measures (MRM) generates recourse directions.

    MRM processes an input dataset for recourse using MRM.process_data(). The
    data processing step can be skipped by directly assigning the class
    attribute _processed_data with data ready for recourse.

    Data provided to _processed_data should
        * Consist only of positively classified examples
        * Not have the label column
        * Be transformed into an embedded numerical space by the adapter.

    Attributes:
        data: The dataset used for recourse.
        adapter: A RecourseAdapter object to transform the data between
            human-readable and embedded space.
        alpha: The alpha function to use during recourse generation.
        rescale_direction: A function for rescaling the recourse.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        adapter: recourse_adapter.RecourseAdapter,
        alpha: AlphaFunction = get_volcano_alpha(),
        rescale_direction: Optional[RecourseRescaler] = normalize_rescaler,
        confidence_threshold: Optional[float] = None,
        model: Optional[model_interface.Model] = None,
        _processed_data: Optional[recourse_adapter.EmbeddedDataFrame] = None,
    ):
        """Creates an MRM instance.

        Args:
            dataset: The dataset to provide recourse over.
            adapter: A RecourseAdapter object to transform the data between
                human-readable and embedded space.
            alpha: The alpha function to use during recourse generation.
            rescale_direction: A function for rescaling the recourse.
            confidence_threshold: If provided, MRM only generates directions
                pointing to areas of sufficiently high model confidence.
            model: Used to check per-datapoint model confidence if
                confidence_threshold is given.
            _processed_data: A processed, recourse-ready dataset to use instead
                of the provided dataset.
        """
        if _processed_data is not None:
            self.data = _processed_data
        else:
            self.data = MRM._process_data(
                dataset,
                adapter,
                confidence_threshold=confidence_threshold,
                model=model,
            )
        self.adapter = adapter
        self.alpha = alpha
        self.rescale_direction = rescale_direction

    @staticmethod
    def _process_data(
        dataset: pd.DataFrame,
        adapter: recourse_adapter.RecourseAdapter,
        confidence_threshold: Optional[float] = None,
        model: Optional[model_interface.Model] = None,
    ) -> recourse_adapter.EmbeddedDataFrame:
        """Processes the dataset for MRM.

        Removes datapoints with negative (-1) labels, drops the class label
        feature column, and transforms the data to a numeric embedded space.
        Also excludes datapoints with below-threshold model confidence.

        Args:
            dataset: The dataset to process.
            adapter: The recourse adapter which transforms data to a numeric
                     embedded space.
            confidence_threshold: If provided, filter out low-confidence ponts.
            model: Used to find low-confidence points if confidence_threshold
                is provided.

        Returns:
            A processed dataset.
        """
        positive_mask = dataset[adapter.label_column] == adapter.positive_label
        # Keep positively-labeled points with sufficiently high model
        # prediction confidence
        if confidence_threshold:
            positive_mask = positive_mask & (
                model.predict_pos_proba(dataset) > confidence_threshold
            )
        positive_dataset = dataset[positive_mask]
        positive_embedded_dataset = adapter.transform(
            positive_dataset.drop(adapter.label_column, axis=1)
        )
        if len(positive_embedded_dataset) == 0:
            raise ValueError(
                "Dataset is empty after excluding negative outcome examples."
            )
        return positive_embedded_dataset

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
        if np.isnan(alpha_val).any():
            raise RuntimeError(
                f"Alpha function returned NaN values: {alpha_val}"
            )
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

    Attributes:
        k_directions: The number of clusters (and recourse directions) to
            generate.
        clusters: The clusters used for generating recourse.
        mrms: The individual per-cluster MRM instances.
    """

    def __init__(
        self,
        k_directions: int,
        adapter: recourse_adapter.RecourseAdapter,
        dataset: pd.DataFrame,
        alpha: AlphaFunction = get_volcano_alpha(),
        rescale_direction: Optional[RecourseRescaler] = normalize_rescaler,
        clusters: Optional[Clusters] = None,
        confidence_threshold: Optional[float] = None,
        model: Optional[model_interface.Model] = None,
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
            confidence_threshold: If provided, MRM only generates directions
                pointing to areas of sufficiently high model confidence.
            model: Used to check per-datapoint model confidence if
                confidence_threshold is given.
        """
        data = MRM._process_data(
            dataset,
            adapter,
            confidence_threshold=confidence_threshold,
            model=model,
        )
        self.k_directions = k_directions
        if not clusters:
            clusters = MRMC._cluster_data(data, self.k_directions)
        self.clusters = clusters
        MRMC._validate_cluster_assignments(
            clusters.cluster_assignments, self.k_directions
        )

        mrms = []
        for cluster_index in range(k_directions):
            cluster_indices = clusters.cluster_assignments[
                clusters.cluster_assignments["datapoint_cluster"]
                == cluster_index
            ]["datapoint_index"]
            data_cluster = data.loc[cluster_indices]
            mrm = MRM(
                dataset=None,  # We provide _processed_data directly.
                adapter=adapter,
                alpha=alpha,
                rescale_direction=rescale_direction,
                _processed_data=data_cluster,
            )
            mrms.append(mrm)
        self.mrms: Sequence[MRM] = mrms

    @staticmethod
    def _cluster_data(
        data: recourse_adapter.EmbeddedDataFrame, k_directions: int
    ) -> Clusters:
        """Clusters the data using sklearn's KMeans clustering.

        Args:
            data: The data to cluster in embedded continuous space.
            k_directions: The number of clusters to generate.
        Returns:
            The data Clusters.
        """
        km = KMeans(n_clusters=k_directions)
        cluster_assignments = km.fit_predict(data)
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

    @staticmethod
    def _validate_cluster_assignments(
        cluster_assignments: pd.DataFrame, k_directions: int
    ) -> bool:
        """Raises an error if the cluster_assignments is invalid.

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


# TODO(@jakeval): Move this somewhere else
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
    expected_directions = expected_directions / np.linalg.norm(
        expected_directions, axis=1
    )
    actual_directions = directions.to_numpy()
    actual_directions = actual_directions / np.linalg.norm(
        actual_directions, axis=1
    )
    return np.diag(expected_directions @ actual_directions.T)
