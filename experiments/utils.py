from typing import Mapping, Sequence, Any, Optional
from sklearn import model_selection
import numpy as np
import pandas as pd


def format_results(
    run_config: Mapping[str, Any], *data_dfs: pd.DataFrame
) -> Sequence[pd.DataFrame]:
    run_id = run_config["run_id"]
    batch_id = run_config["batch_id"]
    result_dfs = []
    for data_df in data_dfs:
        result_df = data_df.copy()
        result_df["run_id"] = run_id
        result_df["batch_id"] = batch_id
        result_dfs.append(result_df)
    index_df = pd.DataFrame(dict((k, [v]) for k, v in run_config.items()))
    return index_df, *result_dfs


def create_run_configs(
    parameter_ranges: Mapping[str, Sequence[Any]],
    num_runs: int = 1,
    random_seed: Optional[int] = None,
) -> Sequence[Mapping[str, Any]]:
    if not random_seed:
        random_seed = np.random.randint(0, 10000)
    run_seeds = np.random.default_rng(random_seed).integers(
        0, 10000, size=num_runs
    )

    # Each test_config is a unique setting over the experimental variables.
    batch_configs = model_selection.ParameterGrid(parameter_ranges)
    run_configs = []
    run_id = 0
    batch_id = 0
    # Each test may have multiple trials, each with a different seed.
    for batch_config in batch_configs:
        for run_seed in run_seeds:
            run_config = {
                "batch_id": batch_id,
                "run_id": run_id,
                "run_seed": int(run_seed),
            }
            run_config.update(batch_config)
            run_configs.append(run_config)
            run_id += 1
        batch_id += 1

    return run_configs
