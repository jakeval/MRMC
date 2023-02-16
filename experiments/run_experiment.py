"""A mainfile for running recourse experiments.

It executes a batch of recourse runs. If the --experiment flag is provided, it
constructs the batch of run configs from an experiment config by performing
grid search over the experiment config parameters.

If the --distributed flag is provided, it uses parallel_runner.py to
split the runs into batches and execute them in parallel. The number of
parallel processes is given by --num_processes.

The recourse method it tests is determined by the --config["recourse_method"]
field."""

# TODO(@jakeval): Clarify experiment terminology
# https://github.com/jakeval/MRMC/issues/41

# TODO(@jakeval): Revisit file structure -- do we still need separate
# directories for dice_experiment, mrmc_experiment, etc?
# https://github.com/jakeval/MRMC/issues/53

import os
import sys
import pathlib

#  Append MRMC/. to the path to fix imports.
_MRMC_PATH = pathlib.Path(os.path.abspath(__file__)).parent.parent
sys.path.append(str(_MRMC_PATH))

from typing import Optional, Mapping, Sequence, Tuple, Any
import shutil
import pathlib
import json
import argparse

from data import data_loader
from data.adapters import continuous_adapter
from models import model_constants
from models import model_loader
from recourse_methods import mrmc_method, dice_method, face_method
from core import recourse_iterator
from core import utils
from experiments import utils as experiment_utils
from experiments import parallel_runner

import numpy as np
import pandas as pd


# MRMC/experiment_results/mrmc_results
_RESULTS_DIR = _MRMC_PATH / "experiment_results"


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


def run_mrmc(
    run_seed: int,
    step_size: float,
    confidence_cutoff: Optional[float],
    noise_ratio: Optional[float],
    rescale_ratio: Optional[float],
    volcano_degree: float,
    volcano_cutoff: float,
    num_clusters: int,
    max_iterations: int,
    dataset_name: str,
    model_type: str,
    cluster_seed: int,
    **_unused_kwargs: Any,
) -> Tuple[Sequence[pd.DataFrame], pd.DataFrame]:
    """Runs MRMC using the given configurations.

    Args:
        run_seed: The seed used in the run. All random numbers are derived from
            this seed except the clustering, which uses cluster_seed.
        step_size: The step size to use when rescaling the recourse.
        confidence_cutoff: The target model confidence.
        noise_ratio: The optional ratio of noise to add.
        rescale_ratio: The optional ratio by which to rescale the direction.
        volcano_degree: The degree to use in the MRM volcano alpha function.
        volcano_cutoff: The cutoff to use in the MRM volcano alpha function.
        num_clusters: The number of clusters to use when generating paths.
        max_iterations: The maximum number of iterations to take recourse steps
            for.
        dataset_name: The name of the dataset to use.
        model_type: The type of model to use.
        cluster_seed: The seed to use for clustering.
        _unused_kwargs: An argument used to capture kwargs from run_config that
            aren't used by this function.

    Returns:
        A list of recourse paths and a dataframe containing cluster info."""
    # generate random seeds
    poi_seed, adapter_seed = np.random.default_rng(run_seed).integers(
        0, 10000, size=2
    )

    # initialize dataset, adapter, model, mrmc, and recourse iterator
    dataset, dataset_info = data_loader.load_data(
        data_loader.DatasetName(dataset_name)
    )
    adapter = continuous_adapter.StandardizingAdapter(
        perturb_ratio=noise_ratio,
        rescale_ratio=rescale_ratio,
        label_column=dataset_info.label_column,
        positive_label=dataset_info.positive_label,
        random_seed=adapter_seed,
    ).fit(dataset)
    model = model_loader.load_model(
        model_constants.ModelType(model_type),
        data_loader.DatasetName(dataset_name),
    )
    mrmc = mrmc_method.MRMC(
        k_directions=num_clusters,
        adapter=adapter,
        dataset=dataset,
        alpha=mrmc_method.get_volcano_alpha(
            cutoff=volcano_cutoff,
            degree=volcano_degree,
        ),
        rescale_direction=mrmc_method.get_constant_step_size_rescaler(
            step_size
        ),
        confidence_threshold=confidence_cutoff,
        model=model,
        random_seed=cluster_seed,
    )
    iterator = recourse_iterator.RecourseIterator(
        adapter=adapter,
        recourse_method=mrmc,
        certainty_cutoff=confidence_cutoff,
        model=model,
    )

    # get the POI
    poi = utils.random_poi(
        dataset,
        label_column=dataset_info.label_column,
        label_value=adapter.negative_label,
        model=model,
        random_seed=poi_seed,
    )

    # generate the paths
    paths = iterator.iterate_k_recourse_paths(
        poi=poi, max_iterations=max_iterations
    )

    # retrieve the clusters
    cluster_df = pd.DataFrame(
        data=mrmc.clusters.cluster_centers,
        columns=adapter.embedded_column_names(),
    )

    return paths, cluster_df


def run_dice(
    run_seed: int,
    confidence_cutoff: Optional[float],
    noise_ratio: Optional[float],
    rescale_ratio: Optional[float],
    num_paths: int,
    max_iterations: int,
    dataset_name: str,
    model_type: str,
    **_unused_kwargs: Any,
) -> Sequence[pd.DataFrame]:
    """Runs DICE using the given configurations.

    Args:
        run_seed: The seed used in the run. All random numbers are derived from
            this seed.
        confidence_cutoff: The target model confidence.
        noise_ratio: The optional ratio of noise to add.
        rescale_ratio: The optional ratio by which to rescale the direction.
        num_paths: The number of paths to generate.
        max_iterations: The maximum number of iterations to take recourse steps
            for.
        dataset_name: The name of the dataset to use.
        model_type: The type of model to use.
        _unused_kwargs: An argument used to capture kwargs from run_config that
            aren't used by this function.

    Returns:
        A list of recourse paths."""
    # generate random seeds
    poi_seed, adapter_seed, dice_seed = np.random.default_rng(
        run_seed
    ).integers(0, 10000, size=3)

    # initialize dataset, adapter, model, dice, and recourse iterator
    dataset, dataset_info = data_loader.load_data(
        data_loader.DatasetName(dataset_name)
    )
    adapter = continuous_adapter.StandardizingAdapter(
        perturb_ratio=noise_ratio,
        rescale_ratio=rescale_ratio,
        label_column=dataset_info.label_column,
        positive_label=dataset_info.positive_label,
        random_seed=adapter_seed,
    ).fit(dataset)
    model = model_loader.load_model(
        model_constants.ModelType(model_type),
        data_loader.DatasetName(dataset_name),
    )
    dice = dice_method.DiCE(
        k_directions=num_paths,
        adapter=adapter,
        dataset=dataset,
        continuous_features=dataset_info.continuous_features,
        desired_confidence=confidence_cutoff,
        model=model,
        random_seed=dice_seed,
    )
    iterator = recourse_iterator.RecourseIterator(
        adapter=adapter,
        recourse_method=dice,
        certainty_cutoff=confidence_cutoff,
        model=model,
    )

    # get the POI
    poi = utils.random_poi(
        dataset,
        label_column=dataset_info.label_column,
        label_value=adapter.negative_label,
        model=model,
        random_seed=poi_seed,
    )

    # generate the paths
    paths = iterator.iterate_k_recourse_paths(
        poi=poi, max_iterations=max_iterations
    )

    return paths


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
    dataset, dataset_info = data_loader.load_data(
        data_loader.DatasetName(dataset_name)
    )
    adapter = continuous_adapter.StandardizingAdapter(
        perturb_ratio=noise_ratio,
        rescale_ratio=rescale_ratio,
        label_column=dataset_info.label_column,
        positive_label=dataset_info.positive_label,
        random_seed=adapter_seed,
    ).fit(dataset)
    model = model_loader.load_model(
        model_constants.ModelType(model_type),
        data_loader.DatasetName(dataset_name),
    )
    face = face_method.FACE(
        dataset=dataset,
        adapter=adapter,
        model=model,
        k_directions=num_paths,
        distance_threshold=distance_threshold,
        confidence_threshold=confidence_cutoff,
        graph_filepath=os.path.join(_MRMC_PATH, graph_filepath),
        counterfactual_mode=counterfactual_mode,
    )
    iterator = recourse_iterator.RecourseIterator(
        adapter=adapter,
        recourse_method=face,
        certainty_cutoff=confidence_cutoff,
        model=model,
    )

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
    path_dfs: Sequence[pd.DataFrame],
    run_config: Mapping[str, Any],
    mrmc_clusters: Optional[pd.DataFrame] = None,
) -> Mapping[str, pd.DataFrame]:
    """Formats the results as DataFrames ready for analysis.

    It adds the path_id and step_id keys to the mrmc_paths dataframe. It also
    adds keys from the experiment_utils.format_results() function.

    Args:
        mrmc_paths: The list of path dataframes output by recourse iteration.
        clusters_df: A dataframe containing cluster information.
        run_config: The run config used to generate these results.

    Returns:
        A mapping from dataframe name to formatted dataframe."""
    for i, path_df in enumerate(path_dfs):
        path_df["step_id"] = np.arange(len(path_df))
        path_df["path_id"] = i
    paths_df = pd.concat(path_dfs).reset_index(drop=True)

    if mrmc_clusters is not None:
        mrmc_clusters["path_id"] = np.arange(len(mrmc_clusters))
        return experiment_utils.format_results(
            run_config, paths_df, mrmc_clusters
        )
    else:
        return experiment_utils.format_results(run_config, paths_df)


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
            _RESULTS_DIR, config["recourse_method"], config["experiment_name"]
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


def _get_results_dir(
    results_directory: Optional[str], config: Mapping[str, Any]
) -> str:
    return results_directory or os.path.join(
        _RESULTS_DIR, config["recourse_method"], config["experiment_name"]
    )


# TODO(@jakeval): The `run` naming here is unclear due to Issue 41
# https://github.com/jakeval/MRMC/issues/41
def run_batch(
    recourse_method: str,
    run_configs: Sequence[Mapping[str, Any]],
    verbose: bool,
) -> Mapping[str, pd.DataFrame]:
    """Executes a batch of runs. Each run is parameterized by a run_config.
    The results of each run are concatenated together in DataFrames and
    returned."""
    all_results = None
    for i, run_config in enumerate(run_configs):
        if recourse_method == "mrmc":
            mrmc_paths, clusters = run_mrmc(**run_config)
            experiment_config_df, mrmc_paths_df, cluster_df = format_results(
                mrmc_paths, run_config, mrmc_clusters=clusters
            )
            run_results = {
                "experiment_config_df": experiment_config_df,
                "mrmc_paths_df": mrmc_paths_df,
                "cluster_df": cluster_df,
            }
        elif recourse_method == "dice":
            experiment_config_df, dice_paths_df = run_dice(**run_config)
            run_results = {
                "experiment_config_df": experiment_config_df,
                "dice_paths_df": dice_paths_df,
            }
        else:
            experiment_config_df, face_paths_df = run_face(**run_config)
            run_results = {
                "experiment_config_df": experiment_config_df,
                "face_paths_df": face_paths_df,
            }
        all_results = merge_results(all_results, run_results)
        if verbose:
            print(f"Finished run {i+1}/{len(run_configs)}")
    return all_results


def get_run_configs(
    config: Mapping[str, Any], is_experiment_config: bool
) -> Sequence[Mapping[str, Any]]:
    """Returns the run configs from the provided config file."""
    if is_experiment_config:
        keys = set(config.keys())
        if keys != set(
            [
                "parameter_ranges",
                "num_runs",
                "random_seed",
                "experiment_name",
                "recourse_method",
            ]
        ) and keys != set(
            [
                "parameter_ranges",
                "num_runs",
                "experiment_name",
                "recourse_method",
            ]
        ):
            raise RuntimeError(
                (
                    "The experiment config should have only the top-level keys"
                    " called 'run_configs', 'num_runs', 'experiment_name', and"
                    f" optionally 'random_seed'. Instead it has keys {keys}."
                )
            )
        return experiment_utils.create_run_configs(
            parameter_ranges=config["parameter_ranges"],
            num_runs=config["num_runs"],
            random_seed=config.get("random_seed", None),
        )
    else:
        keys = set(config.keys())
        if keys != set(["run_configs", "experiment_name", "recourse_method"]):
            raise RuntimeError(
                (
                    "The batch config should only have top-level keys called "
                    "'run_configs' and 'experiment_name'. Instead it has keys "
                    f"{keys}."
                )
            )
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
    recourse_method = config["recourse_method"]
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
            final_results_dir=_get_results_dir(results_dir, config),
            num_processes=num_processes,
            use_slurm=use_slurm,
            recourse_method=recourse_method,
            random_seed=None,  # not needed for reproducibility.
            scratch_dir=scratch_dir,
            verbose=verbose,
        )
        if verbose:
            print(f"Start executing {len(run_configs)} runs.")
        results = runner.execute_runs(run_configs)
    else:
        if verbose:
            print(f"Start executing {len(run_configs)} runs.")
        results = run_batch(recourse_method, run_configs, verbose)
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
