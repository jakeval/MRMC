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

from typing import Sequence, Mapping, Any
import argparse
import subprocess
import json

import numpy as np
import pandas as pd
from sklearn import model_selection


LOCAL_NUM_TRIALS = 5
LOCAL_NUM_PROCESSES = 4
RANDOM_SEED = 1924374


CONFIDENCE_CUTOFFS = [0.5, 0.7]
NOISE_RATIOS = [0]
RESCALE_RATIOS = [1, 0.9, 0.8]
NUM_PATHS = [3]
MAX_ITERATIONS = [30]


STEP_SIZES = [0.5]
VOLCANO_DEGREES = [2]
VOLCANO_CUTOFFS = [0.2]
CLUSTER_SEEDS = [10288294]


DICE_CONFIG = {
    "confidence_cutoff": CONFIDENCE_CUTOFFS,
    "noise_ratio": NOISE_RATIOS,
    "rescale_ratio": RESCALE_RATIOS,
    "num_paths": NUM_PATHS,
    "max_iterations": MAX_ITERATIONS,
}


MRMC_CONFIG = {
    "step_size": STEP_SIZES,
    "confidence_cutoff": CONFIDENCE_CUTOFFS,
    "noise_ratio": NOISE_RATIOS,
    "rescale_ratio": RESCALE_RATIOS,
    "volcano_degree": VOLCANO_DEGREES,
    "volcano_cutoff": VOLCANO_CUTOFFS,
    "num_paths": NUM_PATHS,
    "cluster_seed": CLUSTER_SEEDS,
    "max_iterations": MAX_ITERATIONS,
}


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
parser.add_argument(
    "--local",
    type=bool,
    help="Whether to run locally without SLURM.",
    default=False,
)
parser.add_argument(
    "--method",
    type=str,
    help="Which method to run. dice or mrmc.",
)


def get_params(
    method: str,
    num_trials: int,
) -> Sequence[Mapping[str, Any]]:
    seeds = np.random.default_rng(RANDOM_SEED).integers(10000, size=num_trials)
    if method == "dice":
        base_config = DICE_CONFIG
    elif method == "mrmc":
        base_config = MRMC_CONFIG

    experiment_configs = model_selection.ParameterGrid(base_config)
    experiment_params = []
    trial_id = 0
    test_id = 0
    for experiment_config in experiment_configs:
        for seed in seeds:
            experiment_params.append(
                {
                    "test_id": test_id,
                    "trial_id": trial_id,
                    "seed": int(seed),
                    "recourse_config": experiment_config,
                }
            )
            trial_id += 1
        test_id += 1

    return experiment_params


def partition_trials(trial_param_list, max_processes):
    indices = np.arange(0, len(trial_param_list))
    np.random.default_rng(seed=RANDOM_SEED).shuffle(indices)
    trial_partitions = [
        {
            "trial_param_list": [],
            "process_folder": f"./process_results/results_{i}",
        }
        for i in range(max_processes)
    ]
    partition_idx = 0
    for trial_idx in indices:
        trial_partitions[partition_idx]["trial_param_list"].append(
            trial_param_list[trial_idx]
        )
        partition_idx = (partition_idx + 1) % max_processes

    return trial_partitions


def launch_process(method, trial_param_list, process_folder, run_locally):
    if not os.path.exists(process_folder):
        os.makedirs(process_folder)
    process_config_filename = os.path.join(
        process_folder, "process_config.json"
    )
    process_config = {
        "trial_param_list": trial_param_list,
        "process_folder": process_folder,
    }
    with open(process_config_filename, "w") as f:
        json.dump(process_config, f)

    if method == "dice":
        process_name = "run_dice_trials.py"
    elif method == "mrmc":
        process_name = "run_mrmc_trials.py"
    if not run_locally:
        process_cmd = (
            f"srun -N 1 -n 1 --exclusive python {process_name} "
            f"--config {process_config_filename}"
        )
    else:
        process_cmd = (
            f"python {process_name} --config " f"{process_config_filename}"
        )

    return subprocess.Popen([process_cmd], shell=True), process_folder


def merge_results(method, all_results, process_results_folder):
    def merge_dataframe(all_results, process_df_name):
        df = pd.read_csv(
            os.path.join(process_results_folder, process_df_name + ".csv")
        )
        if not all_results:
            return df
        else:
            return pd.concat([all_results[process_df_name], df]).reset_index(
                drop=True
            )

    if method == "mrmc":
        df_names = ["index_df", "cluster_df", "data_df"]
    elif method == "dice":
        df_names = ["index_df", "data_df"]

    return dict(
        [
            (df_name, merge_dataframe(all_results, df_name))
            for df_name in df_names
        ]
    )


def save_results(df_dict):
    for df_name, df in df_dict.items():
        df.to_csv(f"./{df_name}.csv", index=False)


def run_processes(method, max_processes, trial_param_list, run_locally):
    trial_param_partitions = partition_trials(trial_param_list, max_processes)

    running_processes = {}

    for partition in trial_param_partitions:
        p, process_folder = launch_process(
            method, run_locally=run_locally, **partition
        )
        running_processes[p] = process_folder

    results = None

    while running_processes:
        terminated_processes = []
        for process in running_processes:
            if process.poll() is not None:
                if process.poll() != 0:
                    raise RuntimeError("One of the processes failed!")
                results = merge_results(
                    method, results, running_processes[process]
                )
                terminated_processes.append(process)
        for process in terminated_processes:
            del running_processes[process]

    return results


if __name__ == "__main__":
    args = parser.parse_args()
    trial_param_list = get_params(args.method, args.n_trials)
    if args.max_trials is not None:
        print(f"Generated args for {len(trial_param_list)} trials.")
        trial_param_list = trial_param_list[: args.max_trials]
    print(
        f"Running {len(trial_param_list)} trials across {args.n_procs} tasks."
    )
    if not args.dry_run:
        results = run_processes(
            args.method, args.n_procs, trial_param_list, args.local
        )
        save_results(results)
