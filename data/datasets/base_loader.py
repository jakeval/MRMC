import pandas as pd
from typing import Sequence, Any
from dataclasses import dataclass
import abc
import os
import pathlib


# This is the default directory that data will be saved to and loaded from. It
# evaluates to MRMC/raw_data/.
_DEFAULT_DATA_DIR = (
    pathlib.Path(os.path.normpath(__file__)).parent.parent.parent / "raw_data"
)


@dataclass
class DatasetInfo:
    """An attribute class containing useful info on dataset columns.

    Attributes:
        continuous_features: The names of the dataset's continuous features.
        ordinal_features: The names of the dataset's ordinal features.
        categorical_features: The names of the dataset's categorical features.
        label_name: The name of the dataset's label column.
        positive_label: The label value for positive outcomes."""

    continuous_features: Sequence[str]
    ordinal_features: Sequence[str]
    categorical_features: Sequence[str]
    label_name: str
    positive_label: Any


@dataclass
class DataLoader(abc.ABC):
    """An abstract base class for loading data from the internet or local
    memory.

    If the data is available locally as a csv, it is loaded from disk.
    Otherwise data is downloaded from the internet and saved as a csv.
    Implementing classes should implement download_data and process_data.

    Implementations of this class for a specific dataset are the source of
    truth for domain knowledge about that dataset -- most importantly, they
    record the types of each column.

    Attributes:
        dataset_info: Information about the dataset's columns.
        data_dir: The local directory this dataset is saved to.
        dataset_name: The name of the dataset. Data is saved to
            data_dir/dataset_name/"""

    dataset_info: DatasetInfo
    data_dir: str
    dataset_name: str

    @abc.abstractmethod
    def download_data(self) -> pd.DataFrame:
        """Downloads the data from the internet.

        Returns:
            A DataFrame containing the downloaded data."""

    @abc.abstractmethod
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Processes the raw data.

        Data processing will be different for every dataset. Example operations
        are recategorizing features and dropping rows or columns.

        Args:
            data: The data to process.

        Returns:
            A processed DataFrame."""

    def load_data(self) -> pd.DataFrame:
        """Loads the data from the internet or local disk.

        First checks local disk at self.data_dir. If not saved there, downloads
        the data with self.download_data().

        Returns:
            A DataFrame containing the dataset.
        """
        data_dir = self.data_dir or _DEFAULT_DATA_DIR
        dataset_dir = pathlib.Path(data_dir) / self.dataset_name
        dataset_filepath = dataset_dir / f"{self.dataset_name}.csv"

        if not os.path.exists(dataset_filepath):
            data = self.download_data()
            self.save_data(data, dataset_filepath)
        else:
            data = self.load_local_data(dataset_filepath)

        data = self.process_data(data)
        return data

    def save_data(self, data: pd.DataFrame, dataset_filepath: str) -> None:
        """Saves the data to local disk as a csv.

        Args:
            data: The DataFrame to save.
            dataset_filepath: The filepath to save the data to."""
        if not os.path.exists(pathlib.Path(dataset_filepath).parent):
            os.makedirs(pathlib.Path(dataset_filepath).parent)
        data.to_csv(dataset_filepath, header=True, index=False)

    def load_local_data(self, dataset_filepath: str) -> pd.DataFrame:
        """Loads the data from local disk.

        Args:
            dataset_filepath: The filepath to load the csv data from.

        Returns:
            A DataFrame containing the csv data stored at dataset_filepath."""
        return pd.read_csv(dataset_filepath)
