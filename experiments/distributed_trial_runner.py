import sys
import os

#  Append MRMC/. to the path to fix imports.
sys.path.append(os.path.join(os.getcwd(), ".."))

import tempfile
from typing import Sequence, Mapping, Any, Optional, Tuple
import shutil
import subprocess
import json

import numpy as np
import pandas as pd


class DistributedTrialRunner:
    """Given a python experiment mainfile and a set of trial configs to execute
    it on, run multiple instances of the mainfile over the trial configs in
    parallel and collect their results.

    The mainfile should accept accept the command --config command line
    argument, which is the path to a .json file of trial configs.

    The config.json file is created by the DistributedTrialRunner and contains
    a list of dictionaries mapping configuration variables to values. Each
    dictionary corresponds to one trial.
    """

    def __init__(
        self,
        trial_runner_filename: str,
        final_results_dir: str,
        num_processes: int,
        use_slurm: bool = True,
        random_seed: Optional[int] = None,
        scratch_dir: Optional[str] = None,
    ):
        """Creates a new DistributedTrialRunner.

        Args:
            trial_runner_filename: The name of the python mainfile to run.
            final_results_dir: The directory to write the final results to.
            num_processes: The number of child processes to run in parallel.
            use_slurm: Whether to use the SLURM job scheduler. This should be
                True if you are executing this via the sbatch command.
            random_seed: The random seed to use when assigning trials to
                processes.
            scratch_dir: The directory to use for storing temporary results.
        """
        self.trial_runner_filename = trial_runner_filename
        self.results_dir = final_results_dir
        self.scratch_dir = scratch_dir
        self.num_processes = num_processes
        self.use_slurm = use_slurm
        if not random_seed:
            random_seed = np.random.randint(0, 100000)
        self.rng = np.random.default_rng(random_seed)

    def run_trials(self, trial_configs) -> str:
        partitioned_trial_configs = self.partition_trials(trial_configs)
        with tempfile.TemporaryDirectory(
            prefix=self.scratch_dir
        ) as scratch_dir:
            scratch_results_dir = self.run_processes(
                partitioned_trial_configs, scratch_dir
            )
            self.save_results(scratch_results_dir, self.results_dir)

        return self.results_dir

    def partition_trials(
        self, trial_configs: Sequence[Mapping[str, Any]]
    ) -> Sequence[Sequence[Mapping[str, Any]]]:
        trial_indices = np.arange(0, len(trial_configs))
        self.rng.shuffle(trial_indices)

        # make a config for each process
        partitioned_trial_configs = [[] for _ in range(self.num_processes)]

        # iterate through and partition trial configs into process configs
        partition_idx = 0
        for trial_idx in trial_indices:
            partitioned_trial_configs[partition_idx].append(
                trial_configs[trial_idx]
            )
            partition_idx = (partition_idx + 1) % self.num_processes

        return partitioned_trial_configs

    def run_processes(
        self,
        partitioned_trial_configs: Sequence[Sequence[Mapping[str, Any]]],
        scratch_directory: str,
    ) -> str:
        running_processes = self._launch_processes(
            partitioned_trial_configs, scratch_directory
        )

        return self._collect_process_results(
            running_processes, scratch_directory
        )

    def save_results(
        self, scratch_results_directory: str, final_results_directory: str
    ):
        if os.path.exists(final_results_directory):
            shutil.rmtree(final_results_directory)
        shutil.copytree(scratch_results_directory, final_results_directory)

    def _launch_processes(
        self,
        partitioned_trial_configs: Sequence[Sequence[Mapping[str, Any]]],
        scratch_directory: str,
    ) -> Mapping[subprocess.Popen, str]:
        running_processes: Mapping[subprocess.Popen, str] = {}
        for i, trial_configs in enumerate(partitioned_trial_configs):
            process_io_directory = os.path.join(
                scratch_directory, f"scratch_work_{i}"
            )
            os.mkdir(process_io_directory)
            p, process_results_directory = self._launch_process(
                trial_configs, process_io_directory
            )
            running_processes[p] = process_results_directory
        return running_processes

    def _launch_process(
        self,
        trial_configs: Sequence[Mapping[str, Any]],
        process_io_directory: str,
    ) -> Tuple[subprocess.Popen, str]:
        process_results_directory = os.path.join(
            process_io_directory, "results"
        )
        os.mkdir(process_results_directory)

        process_config_filename = self._write_process_config(
            process_io_directory, trial_configs, process_results_directory
        )

        process_cmd = self._format_subprocess_command(process_config_filename)

        return (
            subprocess.Popen([process_cmd], shell=True),
            process_results_directory,
        )

    def _write_process_config(
        self,
        process_io_directory: str,
        trial_configs: Sequence[Mapping[str, Any]],
        process_results_directory: str,
    ) -> str:
        process_config = {
            "trial_configs": trial_configs,
            "results_directory": process_results_directory,
        }
        process_config_filename = os.path.join(
            process_io_directory, "config.json"
        )
        with open(process_config_filename, "w") as f:
            json.dump(process_config, f)
        return process_config_filename

    def _format_subprocess_command(self, process_config_filename: str) -> str:
        process_cmd = ""
        if self.use_slurm:
            process_cmd = "srun -N 1 -n 1 --exclusive "
        process_cmd += (
            f"python {self.trial_runner_filename} "
            f"--config {process_config_filename}"
        )
        return process_cmd

    def _collect_process_results(
        self,
        running_processes: Mapping[subprocess.Popen, str],
        scratch_directory: str,
    ) -> str:
        # Prepare the aggregated results directory
        aggregated_results_directory = os.path.join(
            scratch_directory, "results"
        )
        os.mkdir(aggregated_results_directory)

        while running_processes:
            # Check for terminated processes and aggregate their results
            terminated_processes = []
            for process in running_processes:
                if process.poll() is not None:
                    if process.poll() != 0:
                        # TODO: make this more descriptive / useful
                        raise RuntimeError("One of the processes failed!")
                    self._aggregate_results(
                        aggregated_results_directory,
                        running_processes[process],
                    )
                    terminated_processes.append(process)
            for process in terminated_processes:
                del running_processes[process]
        return aggregated_results_directory

    def _aggregate_results(
        self, aggregated_results_directory: str, new_results_directory: str
    ) -> None:
        for result in os.listdir(new_results_directory):
            result_path = os.path.join(new_results_directory, result)
            if not os.path.isfile(result_path):
                raise RuntimeError(
                    (
                        f"Result aggregation found directory "
                        f"{result} but only supports .csv files."
                    )
                )
            if not result.endswith(".csv"):
                raise RuntimeError(
                    (
                        f"Result aggregation found file {result} but only "
                        "supports .csv files."
                    )
                )
            new_results = pd.read_csv(result_path)
            aggregated_result_path = os.path.join(
                aggregated_results_directory, result
            )
            if not os.path.exists(aggregated_result_path):
                aggregated_results = new_results
            else:
                old_results = pd.read_csv(aggregated_result_path)
                aggregated_results = pd.concat(
                    [old_results, new_results]
                ).reset_index(drop=True)
            aggregated_results.to_csv(aggregated_result_path)
