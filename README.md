# Monotonic Recourse Measures
Monotonic Recourse Measures (MRM) and Monotonic Recourse Measures with Clustering (MRMC) are methods for Algorithmic Recourse drawing from previous work by Datta et al (1).

The primary contribution of this repository is an implementation of MRM and MRMC. It also includes
* A framework for iterating directional recourse
* Directional implementations of DiCE and FACE to be used as benchmarks
* A set of experiments analyzing the performance of MRMC
* A set of notebooks demonstrating usage of MRMC.


## Contributing
The [Repo overview](#repo-overview) section provides an overview of how the code is structured. If you want to make a change and don't know where to start, it can help point you in the right direction.

For running experiments, the instructions are provided in the `MRMC/experiments/README` document.

For analyzing experiment results, the instructions are provided in the `MRMC/experiment_analysis/README.md` document.

## Immediate next steps
Data for the holistic experiment is fully generated (all methods, models, and datasets), but is not fully analyzed.

* There are analysis notebooks for logistic_regression and credit_card_default in `experiment_analysis`. They should be reviewed -- is there anything missing?
* Analysis notebooks for random_forest and give_me_credit should be created -- the results are still unanalyzed.
* Random perturbation results should be generated and analyzed.

## Repo overview
For an overview of how the experiment infrastructure works, check `MRMC/experiments/README`.

### Key concepts
**Datasets**

The repo provides easy loading and preprocessing of datasets in `MRMC/data`.

**Models**

The repo provides easy loading and training of ML models, especially SKLearn models. To add a new model or adjust an existing one, the most important thing to edit is `models/core/your_model_name.py`. Minor edits (such as adding enum values) will also be necessary in `model_constants.py`, `model_loader.py`, and `scripts/train_model.py`.

As an example of how this all comes together:
* `models/core/logistic_regression.py` extends `models/core/model_trainer.py` and trains an SKLearn logistic regression model.
* All models must follow the `models/model_interface.Model` interface. Logistic regression achieves this by wrapping the SKlearn model in a `models/model_interface.SKLearnModel` class.
* `models/core/model_trainer.py` handles things like saving the logistic regression model.
* The `scripts/train_model.py` file can train logistic regression models from the command line.
* The `models/model_constants.py` and `models/model_loader.py` files allow the rest of the codebase to load the model from memory.

**Dataset adapters**

Recourse adapters handle converting DataFrames and Series' between their native format (potentially with categorical variables) and an embedded format with only continuous features. This is often done automatically, but may occasionally be necessary for the user to handle. When making changes to the codebase, pay close attention to whether you are working with native format DataFrames (pd.DataFrame) or transformed embedded DataFrames (`core/recourse_adapter.EmbeddedDataFrame`). The types should always be annotated to prevent confusion. Note also that labels are also transformed by the adapters.

Recourse adapters also handle converting recourse directions in embedded space into human-readable instructions. Finally, it handles "interpreting" recourse instructions by moving a POI according to the instructions. This typically means adding the recourse direction to the POI in embedded space and then performing an inverse transform back to the native data space.

**Recourse methods**

Recourse methods return recourse directions (in embedded space) and recourse instructions (in human-readable space) for POIs.

### Code structure

**core/**
* Contains code for iterating recourse
* Also contains some miscellaneous functions like direction perturbation and POI selection

**experiment_analysis/**
* Contains notebooks for analyzing experiment results

**experiment_results/**
* Not checked into git
* Contains the raw experiment .csv results

**experiments/**
* Contains code for running experiments

**models/**
* Contains code for training and loading models

**recourse_methods/**
* Contains code implementing (or wrapping) different recourse methods

**scripts/**
* Contains non-experiment executable python mainfiles


(1) https://ieeexplore.ieee.org/document/7546525
