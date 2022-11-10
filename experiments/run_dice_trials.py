import os
import sys

#  Append MRMC/. to the path to fix imports.
sys.path.append(os.path.join(os.getcwd(), ".."))

import json
import argparse

from data import data_loader
from data.adapters import continuous_adapter
from models.core import logistic_regression
from models import model_constants
from recourse_methods import dice_method
from core import recourse_iterator
from core import utils

import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(description="Run an MRMC experiment.")
parser.add_argument(
    "--config",
    type=str,
    help="The filepath of the process config .json.",
)


def save_results(results, process_folder):
    for df_name, df in results.items():
        df_filename = os.path.join(process_folder, df_name + ".csv")
        df.to_csv(df_filename, index=False)


def merge_results(results, trial_results):
    def merge_dataframe(df, trial_df, df_name):
        if not df:
            return trial_df[df_name]
        else:
            return pd.concat([df[df_name], trial_df[df_name]]).reset_index(
                drop=True
            )

    return dict(
        [
            (df_name, merge_dataframe(results, trial_results, df_name))
            for df_name in ["index_df", "data_df"]
        ]
    )


def run_trial(test_id, trial_id, seed, recourse_config):
    dataset, dataset_info = data_loader.load_data(
        data_loader.DatasetName.CREDIT_CARD_DEFAULT
    )
    adapter = continuous_adapter.StandardizingAdapter(
        perturb_ratio=recourse_config["noise_ratio"],
        rescale_ratio=recourse_config["rescale_ratio"],
        label_name=dataset_info.label_name,
        positive_label=dataset_info.positive_label,
    ).fit(dataset)

    model = logistic_regression.LogisticRegression(
        dataset_name=data_loader.DatasetName.CREDIT_CARD_DEFAULT,
        model_name=model_constants.ModelName.DEFAULT,
    ).load_model()

    dice = dice_method.DiCE(
        k_directions=recourse_config["num_paths"],
        adapter=adapter,
        dataset=dataset,
        continuous_features=dataset_info.continuous_features,
        model=model,
        desired_confidence=recourse_config["confidence_cutoff"],
    )

    iterator = recourse_iterator.RecourseIterator(
        adapter=adapter,
        recourse_method=dice,
        certainty_cutoff=recourse_config["confidence_cutoff"],
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
        poi=poi, max_iterations=recourse_config["max_iterations"]
    )

    for i, path in enumerate(paths):
        path["step_id"] = np.arange(len(path))
        path["path_id"] = i
        path["trial_id"] = trial_id
        path["test_id"] = test_id

    index_df = pd.DataFrame(
        {"test_id": [test_id], "trial_id": [trial_id], "seed": [seed]}
    )
    for k, v in recourse_config.items():
        index_df[k] = [v]
    data_df = pd.concat(paths).reset_index(drop=True)
    return {"index_df": index_df, "data_df": data_df}


def run_process(trial_param_list, process_folder):
    results = None
    for trial_params in trial_param_list:
        trial_results = run_trial(**trial_params)
        results = merge_results(results, trial_results)
    save_results(results, process_folder)


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config) as f:
        process_config = json.load(f)
    run_process(**process_config)
