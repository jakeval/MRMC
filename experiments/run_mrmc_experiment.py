import sys
import os

#  Append MRMC/. to the path to fix imports.
sys.path.append(os.path.join(os.getcwd(), ".."))

import argparse

from experiments import utils, distributed_trial_runner


MRMC_MAINFILE = "run_mrmc_trials.py"
FINAL_RESULTS_DIR = "mrmc_results"


# TODO(@jakeval): This should be read from a config
RANDOM_SEED = 1924374  # This is intentionally not a list.
CONFIDENCE_CUTOFFS = [0.5, 0.7]
NOISE_RATIOS = [0]
RESCALE_RATIOS = [1, 0.9, 0.8]
NUM_PATHS = [3]
MAX_ITERATIONS = [30]

STEP_SIZES = [0.5]
VOLCANO_DEGREES = [2]
VOLCANO_CUTOFFS = [0.2]
CLUSTER_SEEDS = [10288294]


EXPERIMENT_CONFIG = {
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
)
parser.add_argument(
    "--n_trials",
    type=int,
    help="The number of trials per test to run.",
)
parser.add_argument(
    "--max_trials",
    type=int,
    help="If provided, only runs up to `max_trials` total.",
    default=None,
)
parser.add_argument(
    "--dry_run",
    help="Whether to actually execute the trials.",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--use_slurm",
    help="Whether to run with the SLURM job scheduler.",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--scratch_dir",
    type=str,
    help="A scratch directory to store temporary results in.",
    default=None,
)
parser.add_argument(
    "--results_dir",
    type=str,
    help="The directory to save the final results to.",
    default=FINAL_RESULTS_DIR,
)


if __name__ == "__main__":
    args = parser.parse_args()
    trial_configs = utils.create_trial_configs(
        EXPERIMENT_CONFIG, args.n_trials, RANDOM_SEED
    )
    print(f"Generated configs for {len(trial_configs)} trials.")
    if not args.dry_run:
        if args.max_trials is not None:
            print(
                f"Throw out all but --max_trials={args.max_trials} "
                "trial configs."
            )
            trial_configs = trial_configs[: args.max_trials]
        print(
            f"Start {len(trial_configs)} trials across {args.n_procs} "
            "tasks..."
        )
        trial_runner = distributed_trial_runner.DistributedTrialRunner(
            trial_runner_filename=MRMC_MAINFILE,
            final_results_dir=args.results_dir,
            num_processes=args.n_procs,
            use_slurm=args.use_slurm,
            scratch_dir=args.scratch_dir,
        )
        final_results_dir = trial_runner.run_trials(trial_configs)
        print(f"Saved results to {final_results_dir}")
