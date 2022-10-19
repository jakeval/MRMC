import pandas as pd
from typing import Sequence, Any
from dataclasses import dataclass
import abc
import os


_DATA_FILENAME = "downloaded_data.csv"
_DATA_PARENT_DIR = "raw_data"


@dataclass
class DataLoader(abc.ABC):
    """An abstract base class for loading data from the internet or local memory.

    If the data is available locally as a csv, it is loaded from disk. Otherwise
    data is downloaded from the internet and saved as a csv. Implementing
    classes should implement download_data and process_data.

    Implementations of this class for a specific dataset are the source of truth
    for domain knowledge about that dataset -- most importantly, they record the
    types of each column.

    Attributes:
        continuous_features: The names of the dataset's continuous features.
        ordinal_features: The names of the dataset's ordinal features.
        categorical_features: The names of the dataset's categorical features.
        label_name: The name of the dataset's label column.
        positive_label: The label value for positive outcomes.
        data_dir: The local directory this dataset is saved to.
    """
    continuous_features: Sequence[str]
    ordinal_features: Sequence[str]
    categorical_features: Sequence[str]
    label_name: str
    positive_label: Any
    data_dir: str

    @abc.abstractmethod
    def download_data(self) -> pd.DataFrame:
        """Downloads the data from the internet.
        
        Returns:
            A DataFrame containing the downloaded data."""
        pass

    @abc.abstractmethod
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Processes the raw data.
        
        Data processing will be different for every dataset. Example operations
        are recategorizing features and dropping rows or columns.
        
        Args:
            data: The data to process.
            
        Returns:
            A processed DataFrame."""
        pass

    def load_data(self) -> pd.DataFrame:
        """Loads the data from the internet or local disk.

        First checks local disk at self.data_dir. If not saved there, downloads
        the data with self.download_data().

        Returns:
            A DataFrame containing the dataset.
        """
        data_dir = os.path.join(self.data_dir, _DATA_PARENT_DIR)
        data_filepath = os.path.join(data_dir, _DATA_FILENAME)

        if not os.path.exists(data_filepath):
            data = self.download_data()
            self.save_data(data, data_filepath)
        else:
            data = self.load_local_data(data_filepath)

        data = self.process_data(data)
        return data

    def save_data(self, data: pd.DataFrame, data_filepath: str) -> None:
        """Saves the data to local disk as a csv.
        
        Args:
            data: The DataFrame to save.
            data_filepath: The filepath to save the data to."""
        data.to_csv(data_filepath, header=True, index=False)

    def load_local_data(self, data_filepath: str) -> pd.DataFrame:
        """Loads the data from local disk.

        Args:
            data_filepath: The filepath to load the csv data from.

        Returns:
            A DataFrame containing the csv data stored at data_filepath."""
        return pd.read_csv(data_filepath)
