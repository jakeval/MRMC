from typing import Mapping, Any
import argparse
import json

import pandas as pd

from models import model_loader


PARSER = argparse.ArgumentParser(description='Train a logistic regression model.')
PARSER.add_argument('--config', type=str, help='Filepath to the config.json file.')


# TODO(@jakeval): This is used by all training files. It should always include
# name, dataset, and type. dataset and type should exist as enums. the git
# commit and file should be automatically added.
def load_config(config_filepath: str) -> Mapping[str, Any]:
    """Loads the config from local disk.
    
    The config is used to drive the experiment and """
    return json.load(config_filepath)


def load_dataset(data_config: Mapping[str, Any]) -> pd.DataFrame:
    pass


if __name__ == '__main__':
    args = PARSER.parse_args()
    config = load_config(args.config)
    dataset = load_dataset(config['data'])
    data_dict = prepare_data(config['data'], config['adapter'])
    model = initialize_model(config['model'])
    history = train_model(model, data_dict, config['training'])
