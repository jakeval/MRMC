"""A simple MRMC experiment to validate efficacy.

* 30 trials
* fixed 0.5 step size
* confidence cutoff 0.75
* random noise varies 0 -> 1
* volcano alpha (degree 2, default cutoff)
* 3 paths

Overview:
For each setting of the variables, run 30 trials.
Each trial is a task. Distribute the tasks amongst the available devices.

Each trial returns a dataframe containing the results.
unique row IDs are generated globally and assigned to as the index to all
dataframes, then these are concatenated.
"""
import sys
import os

#  Append MRMC/. to the path to fix imports.
sys.path.append(os.path.join(os.getcwd(), ".."))

from typing import Sequence, Tuple, Mapping, Any
import argparse
import multiprocessing

import numpy as np
import pandas as pd
from sklearn import model_selection

from recourse_methods import mrmc_method
from core import recourse_iterator, utils
from data import data_loader
from data.adapters import continuous_adapter
from models.core import logistic_regression
from models import model_constants


LOCAL_NUM_TRIALS = 5
LOCAL_NUM_PROCESSES = 4
RANDOM_SEED = 1924374

STEP_SIZES = [0.5]
CONFIDENCE_CUTOFFS = [0.75]
NOISE_RATIOS = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
VOLCANO_DEGREES = [2]
VOLCANO_CUTOFFS = [0.2]
NUM_PATHS = [3]
CLUSTER_SEEDS = [10288294]
MAX_ITERATIONS = [30]


parser = argparse.ArgumentParser(description="Run an MRMC experiment.")
parser.add_argument(
    "--n_procs",
    type=int,
    help="The number of parallel processes to run.",
    default=LOCAL_NUM_PROCESSES,
)
parser.add_argument(
    "--n_trials",
    type=int,
    help="The number of trials per test to run.",
    default=LOCAL_NUM_TRIALS,
)
parser.add_argument(
    "--max_trials",
    type=int,
    help="If provided, only runs up to `max_trials` total.",
    default=None,
)
parser.add_argument(
    "--dry_run",
    type=bool,
    help="Whether to actually execute the trials.",
    default=False,
)


def get_params(
    num_trials: int,
) -> Sequence[Tuple[int, int, int, Mapping[str, Any]]]:
    seeds = np.random.default_rng(RANDOM_SEED).integers(10000, size=num_trials)
    base_config = {
        "step_size": STEP_SIZES,
        "confidence_cutoff": CONFIDENCE_CUTOFFS,
        "noise_ratio": NOISE_RATIOS,
        "volcano_degree": VOLCANO_DEGREES,
        "volcano_cutoff": VOLCANO_CUTOFFS,
        "num_paths": NUM_PATHS,
        "cluster_seed": CLUSTER_SEEDS,
        "max_iterations": MAX_ITERATIONS,
    }

    experiment_configs = model_selection.ParameterGrid(base_config)
    experiment_params = []
    trial_id = 0
    test_id = 0
    for experiment_config in experiment_configs:
        for seed in seeds:
            trial_params = test_id, trial_id, seed, experiment_config
            experiment_params.append(trial_params)
            trial_id += 1
        test_id += 1

    return experiment_params


def run_trial(test_id, trial_id, seed, experiment_config):
    dataset, dataset_info = data_loader.load_data(
        data_loader.DatasetName.CREDIT_CARD_DEFAULT
    )
    adapter = continuous_adapter.StandardizingAdapter(
        perturb_ratio=experiment_config["noise_ratio"],
        label_name=dataset_info.label_name,
        positive_label=dataset_info.positive_label,
    ).fit(dataset)

    model = logistic_regression.LogisticRegression(
        dataset_name=data_loader.DatasetName.CREDIT_CARD_DEFAULT,
        model_name=model_constants.ModelName.DEFAULT,
    ).load_model()

    mrmc = mrmc_method.MRMC(
        k_directions=experiment_config["num_paths"],
        adapter=adapter,
        dataset=dataset,
        alpha=mrmc_method.get_volcano_alpha(
            cutoff=experiment_config["volcano_cutoff"],
            degree=experiment_config["volcano_degree"],
        ),
        rescale_direction=mrmc_method.get_constant_step_size_rescaler(
            experiment_config["step_size"]
        ),
        cluster_seed=seed,
    ).filter_data(experiment_config["confidence_cutoff"], model)

    iterator = recourse_iterator.RecourseIterator(
        adapter=adapter,
        recourse_method=mrmc,
        certainty_cutoff=experiment_config["confidence_cutoff"],
        model=model,
    )

    poi = utils.random_poi(
        dataset,
        column=dataset_info.label_name,
        label=adapter.negative_label,
        seed=seed,
    )

    np.random.seed(seed)

    paths = iterator.iterate_k_recourse_paths(
        poi=poi, max_iterations=experiment_config["max_iterations"]
    )

    for i, path in enumerate(paths):
        path["step_id"] = np.arange(len(path))
        path["path_id"] = i
        path["trial_id"] = trial_id
        path["test_id"] = test_id

    index_df = pd.DataFrame(
        {"test_id": [test_id], "trial_id": [trial_id], "seed": [seed]}
    )
    for k, v in experiment_config.items():
        index_df[k] = [v]
    data_df = pd.concat(paths).reset_index(drop=True)
    return index_df, data_df


def save_dataframes(result_list):
    index_dfs, data_dfs = list(zip(*result_list))
    index_df = pd.concat(index_dfs).reset_index(drop=True)
    data_df = pd.concat(data_dfs).reset_index(drop=True)
    print(data_df)
    print(index_df)
    index_df.to_csv("./index_df.csv")
    data_df.to_csv("./data_df.csv")


if __name__ == "__main__":
    args = parser.parse_args()
    trial_param_list = get_params(args.n_trials)
    if args.max_trials is not None:
        print(f"Generated args for {len(trial_param_list)} trials.")
        trial_param_list = trial_param_list[: args.max_trials]
    print(
        f"Running {len(trial_param_list)} trials across {args.n_procs} tasks."
    )
    if not args.dry_run:
        with multiprocessing.Pool(args.n_procs) as pool:
            results = pool.starmap(run_trial, trial_param_list)
            save_dataframes(results)
