import pandas as pd
from typing import Optional, Mapping, Tuple
from data.datasets import credit_card_default_loader, base_loader
import enum


class DatasetName(enum.Enum):
    """Enum class for dataset names.

    Dataset names are used to identify, load, and save datasets.
    """

    CREDIT_CARD_DEFAULT = credit_card_default_loader.DATASET_NAME


def load_data(
    dataset_name: DatasetName,
    data_dir: Optional[str] = None,
    loader_kwargs: Optional[Mapping] = None,
) -> Tuple[pd.DataFrame, base_loader.DatasetInfo]:
    """Returns the DataFrame and DatasetInfo class for the requested dataset.

    Args:
        dataset_name: The name to identify the dataset.
        data_dir: If provided, where to load data from (or download data to if
            not found). Defaults to MRMC/raw_data/.
        loader_kwargs: Optional key word arguments for the dataset-specific
            loader. For example, the CreditCardDefaultLoader accepts the
            only_continuous argument.

    Returns:
        A DataFrame containing the requested data and a DatasetInfo class
        describing the DataFrame columns.
    """
    loader_kwargs = loader_kwargs or {}
    if dataset_name == DatasetName.CREDIT_CARD_DEFAULT:
        loader = credit_card_default_loader.CreditCardDefaultLoader(
            data_dir=data_dir, **loader_kwargs
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} isn't supported.")
    return loader.load_data(), loader.dataset_info
