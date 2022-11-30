import os
import sys

#  Append MRMC/. to the path to fix imports.
sys.path.append(os.path.join(os.getcwd(), "../.."))

from typing import Optional, Mapping, Sequence, Tuple, Any
import shutil
import pathlib
import json
import argparse

from data import data_loader
from data.datasets import base_loader
from data.adapters import continuous_adapter
from data import recourse_adapter
from models import model_constants
from models import model_interface
from models import model_loader
from recourse_methods import dice_method
from core import recourse_iterator
from core import utils
from experiments import utils as experiment_utils
from experiments import distributed_trial_runner


import numpy as np
import pandas as pd


_RESULTS_DIR = (  # MRMC/experiment_results/mrmc_results
    pathlib.Path(os.path.abspath(__file__)).parent.parent.parent
    / "experiment_results/mrmc_results"
)


parser = argparse.ArgumentParser(description="Run an MRMC experiment.")
parser.add_argument(
    "--config",
    type=str,
    help=(
        "The filepath of the config .json to process and execute. Can be a "
        "batch of run configs or an experiment config is using --experiment."
    ),
)
parser.add_argument(
    "--experiment",
    action="store_true",
    help=(
        "Whether generate a batch of run configs from a single experiment "
        "config .json."
    ),
    default=False,
)
parser.add_argument(
    "--verbose",
    action="store_true",
    default=False,
    help="Whether to print out execution progress.",
)
parser.add_argument(
    "--results_dir",
    type=str,
    help=(
        "The directory to save the results to. Defaults to "
        "MRMC/experiment_results/mrmc_results."
    ),
    default=None,
)
parser.add_argument(
    "--max_runs",
    type=int,
    help="If provided, only runs up to --max_runs total.",
    default=None,
)
parser.add_argument(
    "--dry_run",
    help="If true, generate the run configs but don't execute them.",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--distributed",
    action="store_true",
    default=False,
    help="If true, execute the runs in parallel across -n_procs processes.",
)
parser.add_argument(
    "--n_procs",
    type=int,
    default=None,
    help=(
        "The number of runs to execute in parallel. Required if using "
        "--distributed, otherwise ignored."
    ),
)
parser.add_argument(
    "--slurm",
    action="store_true",
    default=False,
    help=(
        "If true, use SLURM as as distributed job scheduled. Used only if "
        "--distributed is set."
    ),
)
parser.add_argument(
    "--scratch_dir",
    type=str,
    default=None,
    help=(
        "The directory where distributed jobs will write temporary results. "
        "Used only if --distributed is set. Defaults to OS preference."
    ),
)
parser.add_argument(
    "--only_csv",
    action="store_true",
    default=False,
    help=(
        "Save the results as .csv files. This means the .json config file "
        "won't be saved alongside the results."
    ),
)


def _get_dataset(dataset_name) -> Tuple[pd.DataFrame, base_loader.DatasetInfo]:
    """Gets the dataset. Useful for unit testing."""
    return data_loader.load_data(data_loader.DatasetName(dataset_name))


def _get_recourse_adapter(
    dataset: pd.DataFrame,
    dataset_info: base_loader.DatasetInfo,
    random_seed: Optional[int],
    noise_ratio: Optional[float],
    rescale_ratio: Optional[float],
) -> recourse_adapter.RecourseAdapter:
    """Gets the recourse adapter. Useful for unit testing."""
    return continuous_adapter.StandardizingAdapter(
        perturb_ratio=noise_ratio,
        rescale_ratio=rescale_ratio,
        label_column=dataset_info.label_column,
        positive_label=dataset_info.positive_label,
        random_seed=random_seed,
    ).fit(dataset)


def _get_model(model_type: str, dataset_name: str) -> model_interface.Model:
    """Gets the model. Useful for unit testing."""
    return model_loader.load_model(
        model_type=model_constants.ModelType(model_type),
        dataset_name=data_loader.DatasetName(dataset_name),
    )


def _get_dice(
    num_paths: int,
    dataset: pd.DataFrame,
    adapter: recourse_adapter.RecourseAdapter,
    dataset_info: base_loader.DatasetInfo,
    model: model_interface.Model,
    confidence_threshold: float,
    random_seed: int,
    proximity_weight: Optional[float],
    diversity_weight: Optional[float],
    sparsity_weight: Optional[float],
    method: Optional[str],
):
    if (
        (proximity_weight
         or diversity_weight
         or sparsity_weight)
        and not (method == "genetic")
    ):
        raise RuntimeError(
            "DICE optimization weights are only used when method='genetic'"
        )
    dice_kwargs = {}
    if method:
        dice_kwargs["method"] = method

    dice_counterfactual_kwargs = {}
    if proximity_weight:
        dice_counterfactual_kwargs["proximity_weight"] = proximity_weight
    if diversity_weight:
        dice_counterfactual_kwargs["diversity_weight"] = diversity_weight
    if sparsity_weight:
        dice_counterfactual_kwargs["sparsity_weight"] = sparsity_weight

    return dice_method.DiCE(
        k_directions=num_paths,
        adapter=adapter,
        dataset=dataset,
        desired_confidence=confidence_threshold,
        continuous_features=dataset_info.continuous_features,
        model=model,
        dice_kwargs=dice_kwargs,
        dice_counterfactual_kwargs=dice_counterfactual_kwargs,
        random_seed=random_seed,
    )


def _get_recourse_iterator(
    adapter: recourse_adapter.RecourseAdapter,
    dice: dice_method.DiCE,
    confidence_cutoff: float,
    model: model_interface.Model,
) -> recourse_iterator.RecourseIterator:
    """Gets the recourse iterator. Useful for unit testing."""
    return recourse_iterator.RecourseIterator(
        adapter=adapter,
        recourse_method=dice,
        certainty_cutoff=confidence_cutoff,
        model=model,
    )


def run_dice(
    run_config: Mapping[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    run_seed = run_config["run_seed"]
    confidence_cutoff = run_config["confidence_cutoff"]
    noise_ratio = run_config["noise_ratio"]
    rescale_ratio = run_config["rescale_ratio"]
    num_paths = run_config["num_paths"]
    max_iterations = run_config["max_iterations"]
    dataset_name = run_config["dataset"]
    model_type = run_config["model"]
    proximity_weight = run_config.get("proximity_weight", None)
    diversity_weight = run_config.get("diversity_weight", None)
    sparsity_weight = run_config.get("sparsity_weight", None)
    method = run_config.get("method", None)

    # generate random seeds
    dice_seed, poi_seed, adapter_seed = np.random.default_rng(
        run_seed
    ).integers(0, 10000, size=3)

    # initialize dataset, adapter, model, mrmc, and recourse iterator
    dataset, dataset_info = _get_dataset(dataset_name)
    adapter = _get_recourse_adapter(
        dataset=dataset,
        dataset_info=dataset_info,
        random_seed=adapter_seed,
        noise_ratio=noise_ratio,
        rescale_ratio=rescale_ratio,
    )
    model = _get_model(model_type, dataset_name)
    dice = _get_dice(
        num_paths=num_paths,
        dataset=dataset,
        adapter=adapter,
        dataset_info=dataset_info,
        model=model,
        confidence_threshold=confidence_cutoff,
        random_seed=dice_seed,
        proximity_weight=proximity_weight,
        diversity_weight=diversity_weight,
        sparsity_weight=sparsity_weight,
        method=method,
    )
    iterator = _get_recourse_iterator(adapter, dice, confidence_cutoff, model)

    # get the POI
    poi = utils.random_poi(
        dataset,
        label_column=dataset_info.label_column,
        label_value=adapter.negative_label,
        random_seed=poi_seed,
    )

    # generate the paths
    paths = iterator.iterate_k_recourse_paths(
        poi=poi, max_iterations=max_iterations
    )

    return paths


def format_results(
    dice_paths: pd.DataFrame,
    run_config: Mapping[str, Any],
) -> Mapping[str, pd.DataFrame]:
    for i, path in enumerate(dice_paths):
        path["step_id"] = np.arange(len(path))
        path["path_id"] = i
    dice_paths_df = pd.concat(dice_paths).reset_index(drop=True)
    index_df, dice_paths_df = experiment_utils.format_results(
        run_config, dice_paths_df
    )
    return {
        "index_df": index_df,
        "path_df": dice_paths_df,
    }


def merge_results(
    all_results: Optional[Mapping[str, pd.DataFrame]],
    run_results: Mapping[str, pd.DataFrame],
):
    if not all_results:
        return run_results
    else:
        if set(all_results.keys()) != set(run_results.keys()):
            raise RuntimeError(
                (
                    "The new results dictionary contains keys "
                    f"{run_results.keys()}, but the original results have keys"
                    f" {all_results.keys()}"
                )
            )

        merged_results: Mapping[str, pd.DataFrame] = {}
        for result_name in run_results.keys():
            original_result = all_results[result_name]
            new_result = run_results[result_name]
            merged_results[result_name] = pd.concat(
                [original_result, new_result]
            ).reset_index(drop=True)

        return merged_results


# TODO(@jakeval): Unit test this eventually.
def save_results(
    results: Mapping[str, pd.DataFrame],
    results_directory: Optional[str],
    config: Mapping[str, Any],
    only_csv: bool = False,
) -> str:
    if not results_directory:
        results_directory = os.path.join(
            _RESULTS_DIR, config["experiment_name"]
        )
    if os.path.exists(results_directory):
        shutil.rmtree(results_directory)
    os.makedirs(results_directory)
    for result_name, result_df in results.items():
        result_df.to_csv(
            os.path.join(results_directory, result_name + ".csv"), index=False
        )
    if not only_csv:
        with open(
            os.path.join(results_directory, "config.json"), "w"
        ) as config_file:
            json.dump(config, config_file)
    return results_directory


def _get_results_dir(results_directory, experiment_name):
    return results_directory or os.path.join(_RESULTS_DIR, experiment_name)


def run_batch(
    run_configs: Sequence[Mapping[str, Any]],
    verbose: bool,
) -> str:
    all_results = None
    for i, run_config in enumerate(run_configs):
        paths = run_dice(run_config)
        run_results = format_results(paths, run_config)
        all_results = merge_results(all_results, run_results)
        if verbose:
            print(f"Finished run {i+1}/{len(run_configs)}")
    return all_results


def validate_experiment_config(config: Mapping[str, Any]) -> None:
    keys = set(config.keys())
    if keys != set(
        ["parameter_ranges", "num_runs", "random_seed", "experiment_name"]
    ) and keys != set(["parameter_ranges", "num_runs", "experiment_name"]):
        raise RuntimeError(
            (
                "The experiment config should have only the top-level keys "
                "called 'run_configs', 'num_runs', 'experiment_name', and "
                f"optionally 'random_seed'. Instead it has keys {keys}."
            )
        )


def validate_batch_config(config: Mapping[str, Any]) -> None:
    keys = set(config.keys())
    if keys != set(["run_configs", "experiment_name"]):
        raise RuntimeError(
            (
                "The batch config should only have top-level keys called "
                "'run_configs' and 'experiment_name'. Instead it has keys "
                f"{keys}."
            )
        )


def get_run_configs(
    config: Mapping[str, Any], is_experiment_config: bool
) -> Sequence[Mapping[str, Any]]:
    if is_experiment_config:
        validate_experiment_config(config)
        return experiment_utils.create_run_configs(
            parameter_ranges=config["parameter_ranges"],
            num_runs=config["num_runs"],
            random_seed=config.get("random_seed", None),
        )
    else:
        validate_batch_config(config)
        return config["run_configs"]


def main(
    config: Mapping[str, Any],
    is_experiment: bool = False,
    max_runs: Optional[int] = None,
    results_dir: Optional[str] = None,
    verbose: bool = False,
    dry_run: bool = False,
    distributed: bool = False,
    num_processes: Optional[int] = None,
    use_slurm: bool = False,
    scratch_dir: Optional[str] = None,
    only_csv: bool = False,
):
    run_configs = get_run_configs(config, is_experiment)
    if verbose:
        print(f"Got configs for {len(run_configs)} runs.")
    if max_runs:
        run_configs = run_configs[:max_runs]
        if verbose:
            print(f"Throw out all but --max_runs={max_runs} run_configs.")
    if distributed:
        if verbose:
            print(
                f"{len(run_configs)} runs will be distributed over "
                f"{num_processes} processes."
            )
    if not dry_run:
        if distributed:
            runner = distributed_trial_runner.DistributedTrialRunner(
                trial_runner_filename=__file__,
                final_results_dir=_get_results_dir(
                    results_dir, config["experiment_name"]
                ),
                num_processes=num_processes,
                use_slurm=use_slurm,
                random_seed=None,  # not needed for reproducibility.
                scratch_dir=scratch_dir,
                verbose=verbose,
            )
            if verbose:
                print(f"Start executing {len(run_configs)} mrmc runs.")
            results = runner.run_trials(run_configs)
        else:
            if verbose:
                print(f"Start executing {len(run_configs)} mrmc runs.")
            results = run_batch(run_configs, verbose)
        results_dir = save_results(results, results_dir, config, only_csv)
        if verbose:
            print(f"Saved results to {results_dir}")
    elif verbose:
        print("Terminate without executing runs because --dry_run is set.")


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser):
    if args.distributed and not args.n_procs:
        parser.error("--n_procs is required if running with --distributed.")
    if not args.distributed and args.n_procs:
        parser.error("--n_procs is ignored if not running with --distributed.")
    if not args.distributed and args.slurm:
        parser.error("--slurm is ignored if not running with --distributed.")
    if not args.distributed and args.scratch_dir:
        parser.error(
            "--scratch_dir is ignored if not running with --distributed."
        )


if __name__ == "__main__":
    args = parser.parse_args()
    validate_args(args, parser)
    with open(args.config) as config_file:
        config = json.load(config_file)
    main(
        config=config,
        is_experiment=args.experiment,
        max_runs=args.max_runs,
        results_dir=args.results_dir,
        verbose=args.verbose,
        dry_run=args.dry_run,
        distributed=args.distributed,
        num_processes=args.n_procs,
        use_slurm=args.slurm,
        scratch_dir=args.scratch_dir,
        only_csv=args.only_csv,
    )
