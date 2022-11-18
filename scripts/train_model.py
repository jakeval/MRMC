import sys
import os

#  Append MRMC/. to the path to fix imports.
sys.path.append(os.path.join(os.getcwd(), ".."))

import argparse
from models import model_constants
from models.core import model_trainer, logistic_regression
from data import data_loader


parser = argparse.ArgumentParser(description="Train an ML model.")
parser.add_argument("--type", type=str, help="The type of model to train.")
parser.add_argument(
    "--dataset", type=str, help="The name of the dataset to train on."
)
parser.add_argument(
    "--name",
    type=str,
    default=model_constants.ModelName.DEFAULT,
    help="The name of the model to train.",
)


def get_trainer(
    dataset_name: data_loader.DatasetName,
    model_type: model_constants.ModelType,
    model_name: model_constants.ModelName,
) -> model_trainer.ModelTrainer:
    """Creates a ModelTrainer for the requested model and dataset.

    Args:
        dataset_name: The name of the dataset to train on.
        model_type: The type of model family to train.
        model_name: The name of the model to train. Names should be unique
            within within their dataset_name - model_type scope.

    Raises:
        NotImplementedError if the requested ModelType is not supported.

    Returns:
        A ModelTrainer class."""
    if model_type == model_constants.ModelType.LOGISTIC_REGRESSION:
        return logistic_regression.LogisticRegression(
            dataset_name=dataset_name, model_name=model_name
        )
    else:
        raise NotImplementedError(f"Model type {model_type} isn't supported")


if __name__ == "__main__":
    args = parser.parse_args()
    trainer = get_trainer(
        data_loader.DatasetName(args.dataset),
        model_constants.ModelType(args.type),
        model_constants.ModelName(args.name),
    )
    model, results = trainer.new_model()
    print(f"Model achieved results {results}")
