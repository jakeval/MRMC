import pandas as pd
from typing import Optional, Mapping, Tuple, Sequence, Union
from data.datasets import credit_card_default_loader, base_loader
import enum


class DatasetName(enum.Enum):
    """Enum class for dataset names.

    Dataset names are used to identify, load, and save datasets.
    """

    CREDIT_CARD_DEFAULT = credit_card_default_loader.DATASET_NAME
    TOY_DATASET = "toy_data"


def load_data(
    dataset_name: DatasetName,
    data_dir: Optional[str] = None,
    split: Union[str, Sequence[str]] = ["train", "val", "test"],
    loader_kwargs: Optional[Mapping] = None,
) -> Tuple[Union[pd.DataFrame, base_loader.DatasetInfo], ...]:
    """Returns the DataFrame and DatasetInfo class for the requested dataset.

    Args:
        dataset_name: The name to identify the dataset.
        data_dir: If provided, where to load data from (or download data to if
            not found). Defaults to MRMC/raw_data/.
        split: The dataset split to load. Can be any combination of "train",
            "val", or "test".
        loader_kwargs: Optional key word arguments for the dataset-specific
            loader. For example, the CreditCardDefaultLoader accepts the
            only_continuous argument.

    Returns:
        A tuple containing the Dataset split(s) requested and a DatasetInfo
        object containing metadata. The DatasetInfo object is always the last
        element of the tuple.
    """
    loader_kwargs = loader_kwargs or {}
    if dataset_name == DatasetName.CREDIT_CARD_DEFAULT:
        loader = credit_card_default_loader.CreditCardDefaultLoader(
            data_dir=data_dir, **loader_kwargs
        )
    elif dataset_name == DatasetName.TOY_DATASET:
        from confidence_checks import toy_data

        return toy_data.get_data(split)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} isn't supported.")
    return *loader.load_data(split), loader.dataset_info
