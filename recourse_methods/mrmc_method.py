from __future__ import annotations
from typing import Any, Protocol, Sequence
from dataclasses import dataclass
import numpy as np
from core import utils
from models import base_model
import pandas as pd
from data import recourse_adapter
from recourse_methods.base_type import RecourseMethod
from sklearn.cluster import KMeans


class AlphaFunction(Protocol):
    """A data class for alpha functions used by MRM."""
    def alpha(dist: np.ndarray) -> np.ndarray:
        pass


def get_volcano_alpha(cutoff=0.5, degree=2) -> AlphaFunction:
    """Returns a volcano-shaped alpha function.
    
    Args:
        cutoff: The smallest distance after which the alpha output is constant.
        degree: The degree of the exponential."""
    def volcano_alpha(dist: np.ndarray) -> np.ndarray:
        nonlocal cutoff, degree
        return 1/np.where(dist <= cutoff, cutoff, dist)**degree
    return volcano_alpha


class RecourseRescaler(Protocol):
    """A data class for recourse rescaling used by MRM."""
    def rescale(mrm: MRM, dir: pd.DataFrame) -> pd.DataFrame:
        pass


def normalize_rescaler(mrm: MRM, dir: pd.DataFrame) -> pd.DataFrame:
    """Normalizes an MRM direction based on dataset size."""
    return dir / len(mrm.X)


def get_constant_step_size_rescaler(step_size: float = 1):
    """Returns a rescaling function which scales directions to a constant size."""
    return lambda _, dir: utils.constant_step_size(dir, step_size=step_size)


class MRM:
    """Monotone Recourse Measures generates recourse directions."""
    def __init__(self,
                 dataset: pd.DataFrame,
                 adapter: recourse_adapter.RecourseAdapter,
                 label_column: str = 'Y',
                 positive_label: Any = 1,
                 alpha: AlphaFunction = get_volcano_alpha(),
                 rescale_direction: RecourseRescaler = normalize_rescaler):
        """Creates an MRM instance.
        
        Args:
            dataset: The dataset to provide recourse over.
            adapter: The dataset's adapter.
            label_column: The columb providing binary classification outcomes.
            positive_label: The value of a positive classification outcome.
            alpha: The alpha function to use during recourse generation.
            rescale_direction: A function for rescaling the recourse."""
        self.X = MRM.process_data(dataset, adapter, label_column, positive_label)
        self.adapter = adapter
        self.alpha = alpha
        self.rescale_direction = rescale_direction

    @staticmethod
    def process_data(dataset: pd.DataFrame,
                     adapter: recourse_adapter.RecourseAdapter,
                     label_column: str,
                     positive_label: Any) -> recourse_adapter.EmbeddedDataFrame:
        """Processes the dataset for MRM.

        It filters out negative outcomes and transforms the data to a numeric embedded space."""
        positive_dataset = dataset[dataset[label_column] == positive_label]
        X = adapter.transform(positive_dataset.drop(label_column, axis=1))
        if len(X) == 0:
            raise ValueError("Dataset is empty after excluding non-positive outcome examples.")
        return X

    def filter_data(self, confidence_threshold: float, model: base_model.BaseModel) -> MRM:
        """Filters the recourse dataset to include only high-confidence points.

        Args:
            confidence_threshold: All examples that do not achieve at least
                                  this model confidence are removed.
            model: The model to use for generating model confidence.

        Returns:
            Itself. The filtering is done mutably, so the returned version is not a copy."""
        p = model.predict_proba(self.X)
        self.X = self.X[p >= confidence_threshold]
        return self

    def get_unnormalized_direction(self, poi: recourse_adapter.EmbeddedSeries) -> recourse_adapter.EmbeddedSeries:
        """Returns an unnormalized recourse direction."""
        diff = (self.X - poi).to_numpy()
        dist = np.sqrt(np.power(diff, 2).sum(axis=1))
        alpha_val = self.alpha(dist)
        dir = diff.T @ alpha_val
        return recourse_adapter.EmbeddedSeries(index=poi.index, data=dir)

    def get_recourse_direction(self, poi: recourse_adapter.EmbeddedSeries) -> recourse_adapter.EmbeddedSeries:
        """Returns the recourse direction in embedded numeric space."""
        direction = self.get_unnormalized_direction(poi)
        if self.rescale_direction:
            direction = self.rescale_direction(self, direction)
        return direction

    def get_recourse_instructions(self, poi: pd.Series) -> Any:
        """Returns recourse instructions for a given POI."""
        direction = self.get_recourse_direction(self.adapter.transform_series(poi))
        return self.adapter.directions_to_instructions(direction)


@dataclass
class Clusters:
    """Class for kmeans cluster-related data.

    Attributes:
        cluster_assignments: A DataFrame containing columns 'datapoint_index' and 'datapoint_cluster'.
        cluster_centers: A numpy array containing the cluster centers.
    """
    cluster_assignments: pd.DataFrame
    cluster_centers: np.ndarray


class MRMC(RecourseMethod):
    """A class for MRM with clustering."""
    def __init__(self,
                 k_directions: int,
                 adapter: recourse_adapter.RecourseAdapter,
                 dataset: pd.DataFrame,
                 label_column: str = 'Y',
                 positive_label: Any = 1,
                 alpha: AlphaFunction = get_volcano_alpha(),
                 rescale_direction: RecourseRescaler = normalize_rescaler,
                 clusters: Clusters = None):
        """Creates a new MRMC instance.

        MRMC clusters the data and initializes a separate MRM instance for each cluster.
        
        Args:
            k_directions: The number of clusters (and recourse directions) to generate.
            adapter: The dataset adapter.
            dataset: The dataset to perform recourse over.
            label_column: The column containing binary classification outputs.
            positive_label: The value of a positive classification.
            alpha: The alpha function each MRM should use.
            rescale_direction: The rescaling function each MRM should use.
            clusters: The cluster data to use. If None, performs k-means clustering.
        """
        X = MRM.process_data(dataset, adapter, label_column, positive_label)
        self.k_directions = k_directions
        if not clusters:
            clusters = self.cluster_data(X, self.k_directions)
        self.clusters = clusters
        self.validate_cluster_assignments(clusters.cluster_assignments, self.k_directions)

        mrms = []
        for cluster_index in range(k_directions):
            indices = clusters.cluster_assignments[clusters.cluster_assignments['datapoint_cluster'] == cluster_index]['datapoint_index']
            X_cluster = X.loc[indices]
            dataset_cluster = dataset.loc[X_cluster.index]
            mrm = MRM(
                dataset=dataset_cluster,
                adapter=adapter,
                label_column=label_column,
                positive_label=positive_label,
                alpha=alpha,
                rescale_direction=rescale_direction)
            mrms.append(mrm)
        self.mrms: Sequence[MRM] = mrms

    def filter_data(self, confidence_threshold: float, model: base_model.BaseModel) -> MRMC:
        """Filters the recourse dataset to include only high-confidence points.

        Args:
            confidence_threshold: All examples that do not achieve at least
                                  this model confidence are removed.
            model: The model to use for generating model confidence.

        Returns:
            Itself. The filtering is done mutably, so the returned version is not a copy."""
        for mrm in self.mrms:
            mrm.filter_data(confidence_threshold, model)
            return self

    def cluster_data(self, X: recourse_adapter.EmbeddedDataFrame, k_directions: int) -> Clusters:
        """Clusters the data using k-means clustering."""
        km = KMeans(n_clusters=k_directions)
        cluster_assignments = km.fit_predict(X.to_numpy())
        cluster_assignments = np.vstack([X.index.to_numpy(), cluster_assignments]).T
        cluster_assignments_df = pd.DataFrame(columns=['datapoint_index', 'datapoint_cluster'], data=cluster_assignments)
        cluster_centers = km.cluster_centers_
        return Clusters(cluster_assignments=cluster_assignments_df, cluster_centers=cluster_centers)

    def validate_cluster_assignments(self, cluster_assignments: pd.DataFrame, k_directions: int) -> bool:
        """Raises an error if the cluster_assignments is valid.
        
        Invalid cluster_assignments have more or fewer clusters than the requested k_directions.
        """
        cluster_sizes = cluster_assignments.groupby('datapoint_cluster').count()
        if set(cluster_sizes.index) != set(range(k_directions)):
            raise RuntimeError(f'Data was assigned to clusters {cluster_sizes.index}, but expected clusters {range(k_directions)}')
        return True

    def get_all_recourse_directions(self, poi: recourse_adapter.EmbeddedSeries) -> recourse_adapter.EmbeddedDataFrame:
        """Generates different recourse directions for the poi for each of the k_directions."""
        directions = []
        for mrm in self.mrms:
            directions.append(mrm.get_recourse_direction(poi).to_frame().T)
        return pd.concat(directions).reset_index(drop=True)

    def get_all_recourse_instructions(self, poi: pd.Series) -> Sequence[Any]:
        """Generates different recourse instructions for the poi for each of the k_directions."""
        instructions = []
        for mrm in self.mrms:
            instructions.append(mrm.get_recourse_instructions(poi))
        return instructions

    def get_kth_recourse_instructions(self, poi: pd.Series, dir_index: int) -> Any:
        """Generates a single set of recourse instructions for the kth direction."""
        return self.mrms[dir_index].get_recourse_instructions(poi)


def check_dirs(poi: recourse_adapter.EmbeddedSeries, dirs: recourse_adapter.EmbeddedDataFrame, cluster_centers: np.ndarray) -> np.ndarray:
    """Convenience function for verifying the directions generated by MRMC.

    Returns the dot product between the normalized directions and the normalized difference vector
    between the POI and cluster centers.
    """
    expected_dirs = cluster_centers - poi.to_numpy()
    expected_dirs = expected_dirs / np.linalg.norm(expected_dirs, axis=1)[:,None]
    actual_dirs = dirs.to_numpy()
    actual_dirs = actual_dirs / np.linalg.norm(actual_dirs, axis=1)[:,None]
    return np.diag(expected_dirs @ actual_dirs.T)