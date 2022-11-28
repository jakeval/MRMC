from typing import Mapping, Sequence, Any, Optional
from sklearn import model_selection
import numpy as np
import pandas as pd


def format_results(trial_config, *data_dfs):
    trial_id = trial_config["trial_id"]
    test_id = trial_config["test_id"]
    result_dfs = []
    for data_df in data_dfs:
        result_df = data_df.copy()
        result_df["trial_id"] = trial_id
        result_df["test_id"] = test_id
        result_dfs.append(result_df)
    index_df = pd.DataFrame(dict((k, [v]) for k, v in trial_config.items()))
    return index_df, *result_dfs


def create_trial_configs(
    experiment_config: Mapping[str, Sequence[Any]],
    num_trials: int = 1,
    random_seed: Optional[int] = None,
) -> Sequence[Mapping[str, Any]]:
    if not random_seed:
        random_seed = np.random.randint(0, 10000)
    trial_seeds = np.random.default_rng(random_seed).integers(
        0, 10000, size=num_trials
    )

    # Each test_config is a unique setting over the experimental variables.
    test_configs = model_selection.ParameterGrid(experiment_config)
    trial_configs = []
    trial_id = 0
    test_id = 0
    # Each test may have multiple trials, each with a different seed.
    for test_config in test_configs:
        for trial_seed in trial_seeds:
            trial_config = {
                "test_id": test_id,
                "trial_id": trial_id,
                "trial_seed": int(trial_seed),
            }
            trial_config.update(test_config)
            trial_configs.append(trial_config)
            trial_id += 1
        test_id += 1

    return trial_configs
