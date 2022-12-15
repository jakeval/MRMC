from typing import Mapping, Sequence, Any, Optional
from sklearn import model_selection
import numpy as np
import pandas as pd


def format_results(
    run_config: Mapping[str, Any], *data_dfs: pd.DataFrame
) -> Sequence[pd.DataFrame]:
    """Formats result DataFrames with meta data from the run_config.

    It adds a run_id and batch_id column to the dataframes and creates a new
    index dataframe which tracks the run config parameters and ids.

    Args:
        run_config: The config to use when formatting the dataframes.
        data_dfs: The dataframes to format.

    Returns:
        A list of results where the first element is the newly created index_df
        and subsequent results are formatted results DataFrames."""
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
    """Creates a batch of run configs by performing gridsearch over some
    parameter ranges.

    For each parameter combination, a number of run configs equal to num_runs
    is created. Each of these equal-parameter runs shares a batch_id. All runs
    across all batches have a unique run_id. The random seed used for each run
    is unique within a given parameter setting and shared across multiple
    settings. In other words, there are exactly as many run_seeds as there are
    num_runs.

    Args:
        paramter_ranges: The parameters and values to perform gridsearch over.
        num_runs: The number of run configs to create per parameter setting.
        random_seed: The seed used to generate each run_seed.

    Returns:
        A sequence of run configs."""
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
