from typing import Mapping, Sequence
import pandas as pd
import numpy as np


def recategorize_feature(
    column: pd.Series, inverse_category_dict: Mapping[str, Sequence[str]]
) -> pd.Series:
    """Returns a Series where some values are remapped to others.

    Given a dictionary like {'new_val': [val1, val2]}, val1 and val2 are
    relabeled as new_val in the new Series.

    Args:
        column: The series of data to remap.
        inverse_category_dict: The remapping dict formatted as described above.

    Returns:
        A new series with remapped values.
    """
    new_column = column.copy()
    for key, val_list in inverse_category_dict.items():
        for val in val_list:
            new_column = np.where(new_column == val, key, new_column)
    return new_column
