"""Generates and saves a FACE graph for a given dataset."""

import os
import sys

#  Append MRMC/. to the path to fix imports.
sys.path.append(os.path.join(os.getcwd(), ".."))

from typing import Optional
import pathlib
import argparse

from data import data_loader
from data.adapters import continuous_adapter
from recourse_methods import face_method


parser = argparse.ArgumentParser(description="Run an MRMC experiment.")
parser.add_argument(
    "--dataset",
    type=str,
    help=("The dataset to generate a graph for."),
)
parser.add_argument(
    "--distance_threshold",
    type=float,
    help=("The maximum distance to create edges between."),
)
parser.add_argument(
    "--weight_bias",
    type=float,
    default=0,
    help=("The bias to add to each edge weight."),
)
parser.add_argument(
    "--graph_filepath",
    type=str,
    help=("The *.npz filepath to save the graph to."),
)
parser.add_argument(
    "--debug_subsample",
    type=int,
    default=None,
    help=(
        "Optionally, the number of points to sample from the datase when",
        " generating the graph.",
    ),
)


def main(
    dataset_name: str,
    distance_threshold: float,
    weight_bias: float,
    graph_filepath: str,
    debug_subsample: Optional[int],
):
    parent_directory = pathlib.Path(graph_filepath).parent
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)
    dataset, dataset_info = data_loader.load_data(
        data_loader.DatasetName(dataset_name), split="train"
    )
    adapter = continuous_adapter.StandardizingAdapter(
        label_column=dataset_info.label_column,
        positive_label=dataset_info.positive_label,
    ).fit(dataset)
    if debug_subsample:
        dataset = dataset.sample(n=debug_subsample)
    face = face_method.FACE(
        dataset=dataset,
        adapter=adapter,
        model=None,  # doesn't matter
        k_directions=1,  # doesn't matter
        distance_threshold=distance_threshold,
        confidence_threshold=0.6,  # doesn't matter
        weight_bias=weight_bias,
    )
    print("Begin graph generation...")
    face.generate_graph(
        filepath_to_save_to=graph_filepath,
    )
    print("Finished!")


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        args.dataset,
        args.distance_threshold,
        args.weight_bias,
        args.graph_filepath,
        args.debug_subsample,
    )
