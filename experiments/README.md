# Experiment Instructions

## Installing
Note that numpy must be version `1.23.4`.

## Component overview

1. `run_recourse_experiment.py`
    * This mainfile runs all the recourse experiments.
    * Its behavior is determined by config.json files.
    * Experiments are run in sequence on a single process by default.
    * Experiments can be run in parallel by using `--distributed`.
    * Experiments can be run distributed in slurm using `--distributed` and `--slurm`.
    * See [mainfile details](#mainfile-details) for more info.
2. `MRMC/experiments/configs/*`
    * Experiment config file templates are stored here.
    * Each experiment config file template has a 3 versions, one for each of the recourse methods.
    * To run an experiment, the config file parameters must be filled in as described in the [Editing config files](#editing-config-files) section.
3. `run.sbatch`
    * A SLURM sbatch file which takes four arguments. It is usually invoked
        indirectly through `swarm_launcher.sh`.
    * Argument one is the number of processes to distribute the experiment across.
    * Argument two is the directory to write scratch results to.
    * Argument three is a relative path to the experiment config file.
    * Argument four determines the maximum number of runs to execute. It is
        optional.
4. `swarm_launcher.sh`
    * This shell script automates launching the run.sbatch job. It takes three arguments.
    * Argument one is the relative path to the experiment config file.
    * Argument two is the number of total processes to allocate (should be at least 2).
    * Argument three is the maximum number of runs to execute. If not passed, it runs the full experiment.

## Running Locally

A typical single-process experiment invocation looks like

`python run_recourse_experiment.py --config ./your/config/here.json --experiment --verbose`

The experiment behavior is determined by the config.json file passed in with
--config. Experiment configs are stored in 
`MRMC/experiments/configs/[recourse_method]/[config_name].json` and can be
passed in as a relative path. For example, to run StEP hyperparameter tuning,
the command is 
`python run_recourse_experiment.py --config ./configs/mrmc/mrmc_hyperparam.json --experiment --verbose`.

### Running locally for confidence checking

Often running locally is only used for debugging or confidence checking. In this
case, the arguments `--dry_run` and `max_runs N` are useful. A full description
of the arguments is included in [Mainfile details](#mainfile-details)

## Running with --distributed

By default, runs are executed in sequence one after the other. The `--distributed`
flag will distribute runs across `--num_processes` parallel processes. Note that
there is a central orchestrator process, so passing in `--num_processes 3` will
actually result in 4 processes being created (1 controller and 3 children).

## Running on SLURM

The `run.sbatch` and `swarm_launcher.sh` files launch the recourse experiment
mainfile as a slurm job. They launch the process with both `--slurm` and
`--distributed` flags (note that `--slurm` is only necessary if using 
`--distributed`).

For example, to run the StEP hyperparameter tuning with 16 processes, execute
`./swarm_launcher.sh ./configs/mrmc/mrmc_hyperparam.json 16`.

### SLURM resource allocation

SLURM jobs have a time limit set by changing the `#SBATCH --time` field of
`run.sbatch`. Jobs will be cancelled if they run over their allotted time. Jobs
with shorter runtimes or lighter resource requirements will spend less time
waiting in the run queue.

A typical workflow is:
1. Execute `python run_recourse_experiment.py --config ./your/config.json --experiment --verbose --dry_run` locally to check how many runs that config file will attempt to execute.
2. Execute `./swarm_launcher.sh ./your/config.json 2 1` to make sure that a single run can be executed on SLURM without issue.
3. Increase the number of runs (something like 1 -> 50 -> 100 -> 500), increasing the number of processes as necessary to keep the runtime low (5-30 minutes).
4. If you can't scale to more processes (sometimes FACE has issues with memory when using too many processes), increase the expected job runtime.

Useful commands:
* `squeue -u [usernam]` lists your currently running jobs
* `scancel [job id]` cancels a running job
* `sacct -j [job id]` lists information about a job
* `sacct -j [job id] format=Elapsed` lists the runtime of a job

## Experiment infrastructure
The code for running experiments is stored in `experiments/`.

**run_recourse_experiment.py**

This file runs recourse experiments from config files. At a low level, it accepts "batch" config files which contain a list of recourse executions to run. These files are formatted like:
```
{
    "experiment_name": NAME,
    "recourse_method": METHOD,
    "dataset_name": DATASET,
    "split": SPLIT,
    "model_type": MODEL,
    "run_configs": [
        {
            "run_key_1": VALUE1,
            "run_key_2": VALUE2,
            "num_paths": 3  # this is just an example parameter
        },
        {
            "run_key_1": VALUE1,
            "run_key_2": VALUE2,
            "num_paths": 3 
        },
    ]
}
```

Each `run_config` in a batch config file describes the parameters needed to execute a single set of recourse paths for a POI.

In practice, the user will provide "experiment" config files which are automatically processed and turned into a "batch" config file. The process for this is described in [Config files overview](#config-file-conversion) and involves adding some special config parameters (`batch_id`, `run_id`, etc) automatically. These config files look like:
```
{
    "experiment_name": NAME,
    "recourse_method": METHOD,
    "dataset_name": DATASET,
    "split": SPLIT,
    "model_type": MODEL,
    "num_runs": NUM_RUNS,
    "parameter_ranges": {
        "param1": [VAL1, VAL2, VAL3],
        "param2": [VAL1, VAL2]
    }
}
```

After receiving a batch config file with a list of run_configs (or processing an experiment config file into a batch config file), each run_config is executed in sequence using the `run_batch()` function. This function defers generating recourse to the `run_mrmc()`, `run_dice()`, and `run_face()` functions.

**parallel_runner.py**:

The `run_recourse_experiment.py` file always executes run_configs in sequence, so it is quite slow. The `parallel_runner.py` will instead take a list of run_configs, partition them into equally-sized randomly shuffled batches, and spawn a new `run_recoruse_experiment.py` process for each of the batches. In short, it takes a single mega-batch of run_configs and splits into many mini-batches. Each mini-batch is executed by its own process in parallel. A "head" process orchestrates these child process and retrieves their results when they complete.

As an example of how this works:
* The user runs `run_recourse_experiment.py --experiment --config config.json --distributed` with an "experiment" config file.
* The "experiment" config file is turned into a "batch" config file containing a list of run_configs.
* Control is passed from `run_recourse_experiment.py` to `parallel_runner.py` by calling `parallel_runner.execute_runs(run_configs)`
* The `parallel_runner` partitions the run_configs into batches
* The `parallel_runner` creates separate temporary batch config files for each of the partitioned batches and saves them to a partition-dependent scratch directory
    * This scratch directory is where the child `run_recourse_experiment.py` processes will read and write their input and output.
* The `parallel_runner` creates child processes for each run_config batch by calling `run_recourse_experiment.py` with the partition-specific batch config file. If running on slurm (using `--slurm`), the processes are created using `srun`.
* As each process completes, it aggregates and saves their results to a temporary scratch directory.
* Once all child processes complete, it returns the aggregated results as a DataFrame.
* Control is returned to the original `run_recourse_experiment.py` instance along with the final results DataFrame and the final results are saved (either to an automatically decided location or to a user-provided location).

### Config file conversion

See `create_run_configs` in `experiments/utils.py` for implementation.

Config files are converted from "experiment" configs to "batch" configs automatically by performing gridsearch over the values provided in `parameter_ranges`. For example, if there are 3 parameters in `parameter_ranges` and each parameter has two possible values, then a "batch" config with 2 * 3 = 6 run_configs will be created -- there is one run_config for each of the possible combinations of parameter values.

After generating these run_configs, copies of each run_config are created based on the `num_runs` parameter. This parameter allows you to run experiments where there are multiple executions with identical parameters differing only in random seed. If `num_runs = 3`, then for each run_config originally created, two more with identical parameters will be created and added to the final list of run_configs. Some special keys are also added:

* `run_id` is unique to each run_config. If there are 3 original run_configs and `num_runs` is 2, then the final result will have 6 run_configs and 6 unique `run_ids`.
* `batch_id` is shared by all copies of a source run_config. If there are 3 original run_configs and `num_runs` is 2, then the final result will have 6 run_configs and 3 unique `batch_ids`.
* `run_seed` is unique within each batch but shared across all batches. If there are 3 original run_configs and `num_runs` is 2, then the final result will have 6 run_configs and 2 unique `run_seeds`.

Then what does the `run_seed` actually determine? This depends on the implementation of `run_mrmc()`, `run_dice()`, and `run_face()` in `run_recourse_experiment.py`. In practice, the `run_seed` always determines the `adapter_seed` which is used to perturb directions taken by POIs. Additionally if `use_full_eval_set = False`, then the `run_seed` is also used to randomly select a POI from the eval set.

## Editing config files 

The config templates in `MRMC/experiments/configs` should be filled in before running any experiments. The values which must be filled in are called `REPLACE ME`.

The `"dataset_name"` and `"model_type"` parameters must (almost) always be filled in with the appropriate value. `dataset_name` values are `credit_card_default` and `give_me_credit`. `model_type` values are `logistic_regression` and `random_forest`.

The remaining parameters depend on the experiment and are described below.

### Hyperparameter experiment

**MRMC**: No parameters must be filled in since MRMC is only tuned on the credit card default dataset with the logistic regression model.

**DICE**: No parameters must be filled in since DICE is only tuned on the credit card default dataset with the logistic regression model.

**FACE**: FACE should be tuned on all datasets (although the model can be fixed to logistic regression). The most important parameter to tune is `graph_filepath`. Note that graphs of the appropriate distance threshold must first be generated with `scripts/generate_face_graph.py`.

If tuning on `credit_card_default`, then `graph_filepath` should be 
```
"recourse_methods/face_graphs/credit_card_default/graph_0.75.npz",
"recourse_methods/face_graphs/credit_card_default/graph_1.0.npz",
"recourse_methods/face_graphs/credit_card_default/graph_1.5.npz",
"recourse_methods/face_graphs/credit_card_default/graph_2.5.npz",
```

If tuning on `give_me_credit`, then `graph_filepath` should be 
```
"recourse_methods/face_graphs/give_me_credit/graph_0.75.npz",
"recourse_methods/face_graphs/give_me_credit/graph_1.0.npz",
"recourse_methods/face_graphs/give_me_credit/graph_1.5.npz",
```

### Holistic experiment

**MRMC**: No additional parameters must be filled in.

**DiCE**: No additional parameters must be filled in.

**FACE**: If running on `credit_card_default`, the `graph_filepath` should be
`"recourse_methods/face_graphs/credit_card_default/graph_2.5.npz"`.

If running on `give_me_credit`, the `graph_filepath` should be
`"recourse_methods/face_graphs/give_me_credit/graph_1.0.npz"`

## Mainfile details

Output of `python run_recourse_experiment.py -h`:

```
usage: run_recourse_experiment.py [-h] [--config CONFIG] [--experiment] [--verbose]
                                  [--results_dir RESULTS_DIR] [--max_runs MAX_RUNS] [--dry_run]
                                  [--distributed] [--num_processes NUM_PROCESSES] [--slurm]
                                  [--scratch_dir SCRATCH_DIR] [--only_csv]

Run a recourse experiment.

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       The filepath of the config .json to process and execute. Can be a batch of run
                        configs or an experiment config if using --experiment.
  --experiment          Whether to generate a batch of run configs from a single experiment config .json.
  --verbose             Whether to print out execution progress.
  --results_dir RESULTS_DIR
                        The directory to save the results to. Defaults to
                        MRMC/experiment_results/mrmc_results.
  --max_runs MAX_RUNS   If provided, only runs up to --max_runs total.
  --dry_run             If true, generate the run configs but don't execute them.
  --distributed         If true, execute the runs in parallel across -num_processes processes.
  --num_processes NUM_PROCESSES
                        The number of runs to execute in parallel. Required if using --distributed,
                        otherwise ignored.
  --slurm               If true, use SLURM as as distributed job scheduled. Used only if --distributed is
                        set.
  --scratch_dir SCRATCH_DIR
                        The directory where distributed jobs will write temporary results. Used only if
                        --distributed is set. Defaults to OS preference.
  --only_csv            Save the results as .csv files. This means the .json config file won't be saved
                        alongside the results.
```
