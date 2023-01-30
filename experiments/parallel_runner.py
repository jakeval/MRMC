"""Tools for executing multiple recourse experiment runs in parallel.

Given a list of run configs and a python mainfile capable of executing them,
it partitions the run configs into batches and launches a subprocess to
execute each batch."""
import sys
import os

#  Append MRMC/. to the path to fix imports.
sys.path.append(os.path.join(os.getcwd(), ".."))

import tempfile
from typing import Sequence, Mapping, Any, Optional, Tuple
import subprocess
import json

import numpy as np
import pandas as pd


class ParallelRunner:
    """Given a python experiment mainfile and a set of run configs to execute
    it on, run multiple instances of the mainfile over the run configs in
    parallel and collect their results.

    The mainfile should accept accept the command --config command line
    argument, which is the path to a .json file of run configs.

    The config.json file is created by the ParallelRunner and contains
    a list of dictionaries mapping configuration variables to values. Each
    dictionary corresponds to one run.

    Attributes:
        experiment_mainfile_path: The python mainfile to run in parallel.
        final_results_dir: The directory to write the final results to.
        scratch_dir: The process-local directory to write temporary resutls to.
        num_processes: The number of processes to execute in parallel.
        use_slurm: Whether to use the SLURM distributed job scheduler.
        rng: The random generator used to distribute runs across processes.
            It does not effect the results, but may effect load balancing.
        verbose: Whether to print out progress."""

    def __init__(
        self,
        experiment_mainfile_path: str,
        final_results_dir: str,
        num_processes: int,
        use_slurm: bool = True,
        random_seed: Optional[int] = None,
        scratch_dir: Optional[str] = None,
        verbose: bool = False,
    ):
        """Creates a new ParallelRunner.

        Args:
            experiment_mainfile_path: The python mainfile to run in parallel.
            final_results_dir: The directory to write the final results to.
            num_processes: The number of child processes to run in parallel.
            use_slurm: Whether to use the SLURM job scheduler. This should be
                True if you are executing this via the sbatch command.
            random_seed: The random seed to use when assigning runs to
                processes. This has no impact on experiment results.
            scratch_dir: The directory to use for storing temporary results.
        """
        self.experiment_mainfile_path = experiment_mainfile_path
        self.results_dir = final_results_dir
        self.scratch_dir = scratch_dir
        self.num_processes = num_processes
        self.use_slurm = use_slurm
        if not random_seed:
            random_seed = np.random.randint(0, 100000)
        self.rng = np.random.default_rng(random_seed)
        self.verbose = verbose

    def execute_runs(self, run_configs) -> Mapping[str, pd.DataFrame]:
        """Partitions a set of runs into batches and executes them in parallel.

        Each run_config is a dictionary whose format is specified by the
        mainfile which will be executed in parallel.

        Args:
            run_configs: The run configs to partition and execute in parallel.

        Returns:
            A mapping from filename to result DataFrame aggregated across all
            runs."""
        partitioned_run_configs = self.partition_runs(run_configs)
        scratch_dir = ""
        if self.scratch_dir:
            # ensure the scratch_dir is well-formatted.
            scratch_dir = os.path.join(self.scratch_dir, "")
        with tempfile.TemporaryDirectory(prefix=scratch_dir) as scratch_dir:
            scratch_results_dir = self.run_processes(
                partitioned_run_configs, scratch_dir
            )
            return self.collect_results(scratch_results_dir)

    def partition_runs(
        self, run_configs: Sequence[Mapping[str, Any]]
    ) -> Sequence[Sequence[Mapping[str, Any]]]:
        """Randomly partitions a list of run configs into batches to be
        executed in parallel."""
        shuffled_run_indices = np.arange(0, len(run_configs))
        self.rng.shuffle(shuffled_run_indices)

        # initialize a list of empty batches, one for each process
        partitioned_run_configs = [[] for _ in range(self.num_processes)]

        # iterate through and assign run configs into batches configs
        partition_index = 0
        for run_index in shuffled_run_indices:
            partitioned_run_configs[partition_index].append(
                run_configs[run_index]
            )
            partition_index = (partition_index + 1) % self.num_processes

        return partitioned_run_configs

    def run_processes(
        self,
        partitioned_run_configs: Sequence[Sequence[Mapping[str, Any]]],
        scratch_directory: str,
    ) -> str:
        """Runs many processes in parallel where each process is given a
        batch of run configs.

        Each process takes its batch from one of the partitions in the
        partitioned_run_configs argument.

        Args:
            partitioned_run_configs: The list of run config batches to be used
                by each process.
            scratch_directory: The temporary directory each process uses for
                input and output.

        Returns:
            The file directory where .csv results are written."""
        running_processes = self._launch_processes(
            partitioned_run_configs, scratch_directory
        )

        return self._monitor_processes(running_processes, scratch_directory)

    def collect_results(
        self, scratch_results_directory: str
    ) -> Mapping[str, pd.DataFrame]:
        """Loads and returns the final aggregated results from disk."""
        result_dfs = {}
        for result_name in os.listdir(scratch_results_directory):
            result_path = os.path.join(scratch_results_directory, result_name)
            stripped_result_name = os.path.splitext(result_name)[0]
            result_dfs[stripped_result_name] = pd.read_csv(result_path)
        return result_dfs

    def _launch_processes(
        self,
        partitioned_run_configs: Sequence[Sequence[Mapping[str, Any]]],
        scratch_directory: str,
    ) -> Mapping[subprocess.Popen, str]:
        """Asynchronously launches a subprocess for each of the run configs.

        Args:
            partitioned_run_configs: The batches of run configs to use in each
                subprocess.
            scratch_directory: A directory to use write the input and output
                of each subprocess to.

        Returns:
            A mapping from Process objects to the subdirectories they will
            write their results to."""
        running_processes: Mapping[subprocess.Popen, str] = {}
        for i, run_configs in enumerate(partitioned_run_configs):
            process_io_directory = os.path.join(
                scratch_directory, f"scratch_work_{i}"
            )
            os.mkdir(process_io_directory)
            p, process_results_directory = self._launch_process(
                run_configs, process_io_directory
            )
            running_processes[p] = process_results_directory
        return running_processes

    def _launch_process(
        self,
        run_configs: Sequence[Mapping[str, Any]],
        process_io_directory: str,
    ) -> Tuple[subprocess.Popen, str]:
        """Asynchronously launches a subprocess to execute a batch of
        run_configs.

        Args:
            run_configs: The run configs to execute.
            process_io_directory: The directory to use for writing the process
                input and output.

        Returns:
            A tuple of created process and final results path."""
        process_results_directory = os.path.join(
            process_io_directory, "results"
        )
        os.mkdir(process_results_directory)

        process_config_filename = self._write_process_config(
            process_io_directory, run_configs
        )

        process_cmd = self._format_subprocess_command(
            process_config_filename, process_results_directory
        )

        return (
            subprocess.Popen([process_cmd], shell=True),
            process_results_directory,
        )

    def _write_process_config(
        self,
        process_io_directory: str,
        run_configs: Sequence[Mapping[str, Any]],
    ) -> str:
        """Writes the run configs to a config file for the subprocess to read
        and returns the config file path."""
        process_config = {
            "run_configs": run_configs,
            "experiment_name": "experiment_scratch_work",
        }
        process_config_filename = os.path.join(
            process_io_directory, "config.json"
        )
        with open(process_config_filename, "w") as f:
            json.dump(process_config, f)
        return process_config_filename

    def _format_subprocess_command(
        self, process_config_filename: str, results_dir: str
    ) -> str:
        """Formats the command to launch a subprocess.

        If using SLURM, launches the subprocess as an srun step."""
        process_cmd = ""
        if self.use_slurm:
            process_cmd = "srun -N 1 -n 1 --exclusive "
        process_cmd += (
            f"python {self.experiment_mainfile_path} "
            f"--config {process_config_filename} --results_dir {results_dir} "
            "--only_csv"
        )
        return process_cmd

    def _monitor_processes(
        self,
        running_processes: Mapping[subprocess.Popen, str],
        scratch_directory: str,
    ) -> str:
        """Monitors the running subprocesses, aggregating their results as they
        terminate."""
        # Prepare the aggregated results directory
        aggregated_results_directory = os.path.join(
            scratch_directory, "results"
        )
        os.mkdir(aggregated_results_directory)

        if self.verbose:
            print(
                f"Waiting for {len(running_processes)} processes to finish..."
            )

        while running_processes:
            # Check for terminated processes and aggregate their results
            terminated_processes = []
            for process in running_processes:
                if process.poll() is not None:
                    if process.poll() != 0:
                        raise RuntimeError("One of the processes failed!")
                    self._aggregate_results(
                        aggregated_results_directory,
                        running_processes[process],
                    )
                    terminated_processes.append(process)
            for process in terminated_processes:
                del running_processes[process]
                if self.verbose:
                    remaining_procs = len(running_processes)
                    print(
                        f"Waiting for {remaining_procs} processes to finish..."
                    )
        return aggregated_results_directory

    def _aggregate_results(
        self, aggregated_results_directory: str, new_results_directory: str
    ) -> None:
        """Concatenates the newly generated results with the previous results.

        The newly merged results are saved over the original aggregated
        results."""
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
            aggregated_results.to_csv(aggregated_result_path, index=False)
