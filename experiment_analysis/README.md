# Experiment Analysis Instructions

## Component overview

1. Analysis notebook templates
    * Under `experiment_analysis`, there are `*_template.ipynb` template notebooks for each experiment.
    * Template notebooks are named after the experiment they analyze, so a template for analying the holistic results would be named `holistic_template.ipynb`.
    * When using a template analysis notebook to analyze the results from a given recourse method, dataset, and model type, the template should be copied to a new location and its **TODO** variables should be filled in with appropriate values. See [Template usage](#template-usage) for details.
2. Analysis notebooks
    * These are the analysis notebooks (typically derived from analysis templates) which analyze the experiment results.
3. Cached results
    * Because analyzing the results takes a lot of time, the analyzed results are automatically cached. If you wish to re-analyze the results without using cached values, set `USE_CACHED_RESULTS` to False.

## Template usage
Each template has usage instructions guided by **TODO**s in the Markdown cells. To use a template notebook, it should be copied to a new directory dependent on which recourse method, model type, and dataset you wish to analzye. The directories are constructed as `experiment_analysis/recourse_method/model_type/dataset_name/recourse_name_experiment_name.ipynb`. So to analyze MRMC holistic results using the logistic regression model on the credit card default dataset, you would copy the `holistic_template.ipynb` to `experiment_analysis/mrmc/logistic_regression/credit_card_default/mrmc_holistic.ipynb`.

## Running on SWARM.

Based on https://shreyas269.github.io/posts/2012/08/blog-post-4/.

1. SSH into Swarm.
2. Run `srun --nodes=1 --ntasks-per-node=1 --time=01:00:00 --pty bash -i`.
    * This starts a 1 hour session on a compute node.
3. In the compute node, run `ssh -R 2222:localhost:2222 swarm.cs.umass.edu -f -N`
4. In the compute node, activate the virtual environment and run `python -m ipykernel install --user --name venv`
5. In the compute node, start the jupyter server with `jupyter notebook --port 2222 --no-browser`
6. On your local machine, run `ssh -L 2222:localhost:2222 <user>@swarm.cs.umass.edu -f -N`
    * I think `-f` makes it run in the background.
    * I don't know what `-N` does.
7. On your local machine, use a browser to visit `localhost:2222`.
8. Make sure the jupyter kernel is set to `venv`.