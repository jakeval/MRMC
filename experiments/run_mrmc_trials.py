import os
import sys

#  Append MRMC/. to the path to fix imports.
sys.path.append(os.path.join(os.getcwd(), ".."))

from typing import Optional
import json
import argparse
import dataclasses

from data import data_loader
from data.adapters import continuous_adapter
from models.core import logistic_regression
from models import model_constants
from recourse_methods import mrmc_method
from core import recourse_iterator
from core import utils
from experiments import utils as experiment_utils

import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(description="Run an MRMC experiment.")
parser.add_argument(
    "--config",
    type=str,
    help="The filepath of the process config .json.",
)


# TODO(@jakeval): How should everything be split across files?
# TODO(@jakeval): how should test_id and trial_id be handled?


@dataclasses.dataclass
class TrialResults:
    index_df: pd.DataFrame
    cluster_df: pd.DataFrame
    path_df: pd.DataFrame


def run_trial(trial_config):
    trial_seed = trial_config["trial_seed"]
    step_size = trial_config["step_size"]
    confidence_cutoff = trial_config["confidence_cutoff"]
    noise_ratio = trial_config["noise_ratio"]
    rescale_ratio = trial_config["rescale_ratio"]
    volcano_degree = trial_config["volcano_degree"]
    volcano_cutoff = trial_config["volcano_cutoff"]
    num_paths = trial_config["num_paths"]
    max_iterations = trial_config["max_iterations"]

    mrmc_seed, poi_seed, adapter_seed = np.random.default_rng(
        trial_seed
    ).integers(0, 10000, size=3)

    dataset, dataset_info = data_loader.load_data(
        data_loader.DatasetName.CREDIT_CARD_DEFAULT
    )
    adapter = continuous_adapter.StandardizingAdapter(
        perturb_ratio=noise_ratio,
        rescale_ratio=rescale_ratio,
        label_column=dataset_info.label_column,
        positive_label=dataset_info.positive_label,
        random_seed=adapter_seed,
    ).fit(dataset)

    model = logistic_regression.LogisticRegression(
        dataset_name=data_loader.DatasetName.CREDIT_CARD_DEFAULT,
        model_name=model_constants.ModelName.DEFAULT,
    ).load_model()

    mrmc = mrmc_method.MRMC(
        k_directions=num_paths,
        adapter=adapter,
        dataset=dataset,
        alpha=mrmc_method.get_volcano_alpha(
            cutoff=volcano_cutoff,
            degree=volcano_degree,
        ),
        rescale_direction=mrmc_method.get_constant_step_size_rescaler(
            step_size
        ),
        random_seed=mrmc_seed,
        confidence_threshold=confidence_cutoff,
        model=model,
    )

    clusters = mrmc.clusters.cluster_centers
    cluster_df = pd.DataFrame(
        data=clusters,
        columns=adapter.embedded_column_names(),
    )

    iterator = recourse_iterator.RecourseIterator(
        adapter=adapter,
        recourse_method=mrmc,
        certainty_cutoff=confidence_cutoff,
        model=model,
    )

    poi = utils.random_poi(
        dataset,
        label_column=dataset_info.label_column,
        label_value=adapter.negative_label,
        seed=poi_seed,
    )

    paths = iterator.iterate_k_recourse_paths(
        poi=poi, max_iterations=max_iterations
    )

    return paths, cluster_df


def format_results(paths, cluster_df, trial_config):
    cluster_df["path_id"] = np.arange(len(cluster_df))
    for i, path in enumerate(paths):
        path["step_id"] = np.arange(len(path))
        path["path_id"] = i
    path_df = pd.concat(paths).reset_index(drop=True)
    index_df, cluster_df, path_df = experiment_utils.format_results(
        trial_config, cluster_df, path_df
    )
    return TrialResults(index_df, cluster_df, path_df)


def merge_results(
    results: Optional[TrialResults], trial_results: TrialResults
):
    if not results:
        return trial_results
    else:
        return TrialResults(
            index_df=pd.concat(
                [results.index_df, trial_results.index_df]
            ).reset_index(drop=True),
            cluster_df=pd.concat(
                [results.cluster_df, trial_results.cluster_df]
            ).reset_index(drop=True),
            path_df=pd.concat(
                [results.path_df, trial_results.path_df]
            ).reset_index(drop=True),
        )


def save_results(results: TrialResults, results_directory):
    index_df_path = os.path.join(results_directory, "index_df.csv")
    cluster_df_path = os.path.join(results_directory, "cluster_df.csv")
    path_df_path = os.path.join(results_directory, "path_df.csv")

    results.index_df.to_csv(index_df_path, index=False)
    results.cluster_df.to_csv(cluster_df_path, index=False)
    results.path_df.to_csv(path_df_path, index=False)


def run_process(trial_configs, results_directory):
    all_results = None
    for trial_config in trial_configs:
        paths, clusters = run_trial(trial_config)
        trial_results = format_results(paths, clusters, trial_config)
        all_results = merge_results(all_results, trial_results)
    save_results(all_results, results_directory)


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config) as f:
        process_config = json.load(f)
    run_process(**process_config)
