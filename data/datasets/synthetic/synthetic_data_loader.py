import pandas as pd
import numpy as np
from typing import Any, Tuple
from data import data_preprocessor as dp


def load_data() -> Tuple[pd.DataFrame, dp.Preprocessor]:
    pc1 = np.random.normal((-3, 2), 1, size=(100, 2))
    pc2 = np.random.normal((0, 2), 1, size=(100, 2))
    pc3 = np.random.normal((3, 2), 1, size=(100, 2))
    nc1 = np.random.normal((-1, -1), 0.75, size=(150, 2))
    nc2 = np.random.normal((1, -1), 0.75, size=(150, 2))
    data = np.concatenate([pc1, pc2, pc3, nc1, nc2])
    labels = [1] * 300 + [-1] * 300

    df = pd.DataFrame(columns=['x', 'y'], data=data)
    df['Y'] = labels

    return df, dp.PassthroughPreprocessor().fit(df)
