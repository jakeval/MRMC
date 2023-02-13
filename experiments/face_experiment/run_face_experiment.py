"""A mainfile for running FACE experiments.

It executes a batch of FACE runs. If the --experiment flag is provided, it
constructs the batch of run configs from an experiment config by performing
grid search over the experiment config parameters.

If the --distributed flag is provided, it uses parallel_runner.py to
split the runs into batches and execute them in parallel. The number of
parallel processes is given by --num_processes."""

import os
import sys
import pathlib

#  Append MRMC/. to the path to fix imports.
mrmc_path = pathlib.Path(os.path.abspath(__file__)).parent.parent.parent
sys.path.append(str(mrmc_path))

from typing import Optional, Mapping, Sequence, Tuple, Any
import shutil
import pathlib
import json
import argparse
import time

from data import data_loader
from data.datasets import base_loader
from data.adapters import continuous_adapter
from data import recourse_adapter
from models import model_constants
from models import model_interface
from models import model_loader
from recourse_methods import face_method
from core import recourse_iterator
from core import utils
from experiments import utils as experiment_utils
from experiments import parallel_runner


import numpy as np
import pandas as pd


# TODO(@jakeval): https://github.com/jakeval/MRMC/issues/44

_RESULTS_DIR = (  # MRMC/experiment_results/face_results
    pathlib.Path(os.path.abspath(__file__)).parent.parent.parent
    / "experiment_results/face_results"
)

# The directory to the MRMC repo this file is in.
# It is used to retrieve FACE graphs.
_MRMC_DIR = pathlib.Path(os.path.abspath(__file__)).parent.parent.parent


parser = argparse.ArgumentParser(description="Run a FACE experiment.")
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
        "MRMC/experiment_results/face_results."
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
    help=(
        "If true, execute the runs in parallel across -num_processes "
        "processes."
    ),
)
parser.add_argument(
    "--num_processes",
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


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser):
    """Validates the command line args.

    If the --distributed flag is provided without the --num_processes flag, an
    error is raised. If the --num_processes, --slurm, or --scratch_dir args are
    provided without the --distributed flag, an error is raised."""
    if args.distributed and not args.num_processes:
        parser.error(
            "--num_processes is required if running with --distributed."
        )
    if not args.distributed and args.num_processes:
        parser.error(
            "--num_processes is ignored if not running with --distributed."
        )
    if not args.distributed and args.slurm:
        parser.error("--slurm is ignored if not running with --distributed.")
    if not args.distributed and args.scratch_dir:
        parser.error(
            "--scratch_dir is ignored if not running with --distributed."
        )


def _get_dataset(
    dataset_name: str,
) -> Tuple[pd.DataFrame, base_loader.DatasetInfo]:
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


def _get_face(
    dataset: pd.DataFrame,
    adapter: recourse_adapter.RecourseAdapter,
    model: model_interface.Model,
    num_paths: int,
    confidence_threshold: float,
    distance_threshold: float,
    graph_filepath: str,
    counterfactual_mode: bool,
) -> face_method.FACE:
    """Gets the FACE instance. Useful for unit testing."""
    full_graph_filepath = os.path.join(_MRMC_DIR, graph_filepath)
    return face_method.FACE(
        dataset=dataset,
        adapter=adapter,
        model=model,
        k_directions=num_paths,
        distance_threshold=distance_threshold,
        confidence_threshold=confidence_threshold,
        graph_filepath=full_graph_filepath,
        counterfactual_mode=counterfactual_mode,
    ).fit()


def _get_recourse_iterator(
    adapter: recourse_adapter.RecourseAdapter,
    face: face_method.FACE,
    confidence_cutoff: float,
    model: model_interface.Model,
) -> recourse_iterator.RecourseIterator:
    """Gets the recourse iterator. Useful for unit testing."""
    return recourse_iterator.RecourseIterator(
        adapter=adapter,
        recourse_method=face,
        certainty_cutoff=confidence_cutoff,
        model=model,
    )


def run_face(
    run_seed: int,
    confidence_cutoff: Optional[float],
    noise_ratio: Optional[float],
    rescale_ratio: Optional[float],
    num_paths: int,
    max_iterations: int,
    dataset_name: str,
    model_type: str,
    distance_threshold: float,
    graph_filepath: str,
    counterfactual_mode: bool,
    **_unused_kwargs: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Runs FACE using the given configurations.

    Args:
        run_seed: The seed used in the run. All random numbers are derived from
            this seed except the clustering, which uses cluster_seed.
        confidence_cutoff: The target model confidence.
        noise_ratio: The optional ratio of noise to add.
        rescale_ratio: The optional ratio by which to rescale the direction.
        num_paths: The number of recourse paths to generate.
        max_iterations: The maximum number of iterations to take recourse steps
            for.
        dataset_name: The name of the dataset to use.
        model_type: The type of model to use.
        distance_threshold: The maximum edge length of the graph.
        graph_filepath: Path to a graph which matches the distance_threshold.
        counterfactual_mode: Whether to use the first or final point in the
            path to create recourse directions.
        _unused_kwargs: An argument used to capture kwargs from run_config that
            aren't used by this function.

    Returns:
        A list of recourse paths and a dataframe containing cluster info."""
    # generate random seeds
    poi_seed, adapter_seed = np.random.default_rng(run_seed).integers(
        0, 10000, size=2
    )

    # initialize dataset, adapter, model, face, and recourse iterator
    dataset, dataset_info = _get_dataset(dataset_name)
    adapter = _get_recourse_adapter(
        dataset=dataset,
        dataset_info=dataset_info,
        random_seed=adapter_seed,
        noise_ratio=noise_ratio,
        rescale_ratio=rescale_ratio,
    )
    model = _get_model(model_type, dataset_name)
    face = _get_face(
        dataset=dataset,
        adapter=adapter,
        model=model,
        num_paths=num_paths,
        confidence_threshold=confidence_cutoff,
        distance_threshold=distance_threshold,
        graph_filepath=graph_filepath,
        counterfactual_mode=counterfactual_mode,
    )
    iterator = _get_recourse_iterator(adapter, face, confidence_cutoff, model)

    # get the POI
    poi = utils.random_poi(
        dataset,
        label_column=dataset_info.label_column,
        label_value=adapter.negative_label,
        random_seed=poi_seed,
        model=model,
    )

    # generate the paths
    paths = iterator.iterate_k_recourse_paths(
        poi=poi, max_iterations=max_iterations
    )

    return paths


def format_results(
    face_paths: pd.DataFrame,
    run_config: Mapping[str, Any],
) -> Mapping[str, pd.DataFrame]:
    """Formats the results as DataFrames ready for analysis.

    It adds the path_id and step_id keys to the face_paths dataframe. It also
    adds keys from the experiment_utils.format_results() function.

    Args:
        mrmc_paths: The list of path dataframes output by recourse iteration.
        run_config: The run config used to generate these results.

    Returns:
        A mapping from dataframe name to formatted dataframe."""
    for i, path in enumerate(face_paths):
        path["step_id"] = np.arange(len(path))
        path["path_id"] = i
    face_paths_df = pd.concat(face_paths).reset_index(drop=True)
    experiment_config_df, face_paths_df = experiment_utils.format_results(
        run_config, face_paths_df
    )
    return {
        "experiment_config_df": experiment_config_df,
        "face_paths_df": face_paths_df,
    }


def merge_results(
    all_results: Optional[Mapping[str, pd.DataFrame]],
    run_results: Mapping[str, pd.DataFrame],
):
    """Concatenates newly generated result dataframes with previously generated
    result dataframes.

    Args:
        all_results: The previously generated results.
        run_results: The newly generated results to merge into the previous
            results.

    Returns:
        A merged set of results containing data from all_results and
        run_results."""
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
    """Saves the results and experiment config to the local file system.

    Args:
        results: A mapping from filename to DataFrame.
        results_directory: Where to save the results.
        config: The config used to run this experiment.

    Returns:
        The directory where the results are saved."""
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
) -> Mapping[str, pd.DataFrame]:
    """Executes a batch of runs. Each run is parameterized by a run_config.
    The results of each run are concatenated together in DataFrames and
    returned."""
    all_results = None
    for i, run_config in enumerate(run_configs):
        paths = run_face(**run_config)
        run_results = format_results(paths, run_config)
        all_results = merge_results(all_results, run_results)
        if verbose:
            print(f"Finished run {i+1}/{len(run_configs)}")
    return all_results


def validate_experiment_config(config: Mapping[str, Any]) -> None:
    """Validates the formatting of an experiment config."""
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
    """Validates the formatting of a batch config."""
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
    """Returns the run configs from the provided config file."""
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


def do_dry_run(
    config: Mapping[str, Any],
    is_experiment: bool = False,
    max_runs: Optional[int] = None,
):
    run_configs = get_run_configs(config, is_experiment)
    print(f"Got configs for {len(run_configs)} runs.")
    if max_runs:
        run_configs = run_configs[:max_runs]
        print(f"Throw out all but --max_runs={max_runs} run_configs.")
    print("Terminate without executing runs because --dry_run is set.")


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
    if dry_run:
        do_dry_run(config, is_experiment, max_runs)
        return
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
        runner = parallel_runner.ParallelRunner(
            experiment_mainfile_path=__file__,
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
        results = runner.execute_runs(run_configs)
    else:
        if verbose:
            print(f"Start executing {len(run_configs)} mrmc runs.")
        results = run_batch(run_configs, verbose)
    results_dir = save_results(results, results_dir, config, only_csv)
    if verbose:
        print(f"Saved results to {results_dir}")


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
        num_processes=args.num_processes,
        use_slurm=args.slurm,
        scratch_dir=args.scratch_dir,
        only_csv=args.only_csv,
    )
