import sys
import os

#  Append MRMC/. to the path to fix imports.
sys.path.append(os.path.join(os.getcwd(), ".."))

from typing import Optional
import argparse
from sklearn import model_selection, neighbors
import joblib
import numpy as np
import logging

from data import data_loader
from data.adapters import continuous_adapter


_RANDOM_SEED = 102943
N_SPLITS = 5
BANDWIDTHS = np.logspace(-1, 0, 6)

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Fit a KDE model.")
parser.add_argument(
    "--dataset", type=str, help="The name of the dataset to train on."
)
parser.add_argument(
    "--kde_directory",
    type=str,
    help="The directory to save the KDE to.",
    default=None,
)


def fit_kde(
    dataset_name: str, kde_directory: Optional[str] = None
) -> neighbors.KernelDensity:
    """Fit a KDE over the dataset using 5-fold cross validation over the
    training set.

    Bandwidth is chosen from the range np.logspace(-1, 0, 6).

    Args:
        dataset_name: The dataset to fit the KDE over.
        kde_directory: If provided, the directory to save the KDE to.

    Returns:
        A fit Kernel Density Estimator."""

    dataset, dataset_info = data_loader.load_data(
        data_loader.DatasetName(dataset_name), split="train"
    )

    rng = np.random.RandomState(_RANDOM_SEED)

    adapter = continuous_adapter.StandardizingAdapter(
        label_column=dataset_info.label_column,
        positive_label=dataset_info.positive_label,
    ).fit(dataset)

    kfold = model_selection.KFold(n_splits=N_SPLITS)

    transformed_data = adapter.transform(
        dataset.drop(columns=dataset_info.label_column)
    ).sample(frac=1, replace=False, random_state=rng)

    scores = []

    for bandwidth in BANDWIDTHS:
        # logging for status monitoring
        logging.info(f"Evaluate bandwidth {bandwidth}.")
        score = 0
        for train_indices, val_indices in kfold.split(transformed_data):
            kde = neighbors.KernelDensity(bandwidth=bandwidth).fit(
                transformed_data.iloc[train_indices]
            )
            score += kde.score(transformed_data.iloc[val_indices])
        scores.append(score / len(BANDWIDTHS))
        logging.info(f"Bandwidth {bandwidth} has score {scores[-1]}")

    best_bandwidth = BANDWIDTHS[np.argmax(scores)]
    # printing for ordinary command script usage
    print(f"Selected bandwidth is {best_bandwidth}.")
    kde = neighbors.KernelDensity(bandwidth=best_bandwidth).fit(
        transformed_data
    )
    if kde_directory:
        kde_filepath = os.path.join(
            kde_directory, f"{dataset_name}_kde.joblib"
        )
        joblib.dump(kde, kde_filepath)
        print(f"Saved KDE to {kde_filepath}.")
    return kde


if __name__ == "__main__":
    args = parser.parse_args()
    fit_kde(args.dataset, args.kde_directory)
