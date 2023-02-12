import pathlib
import os
import enum


#  The default path for saved models is MRMC/saved_models/
MODEL_DIR = (
    pathlib.Path(os.path.normpath(__file__)).parent.parent / "saved_models"
)


RANDOM_SEED = 12348571  # Random seed to use throughout model training


class ModelType(enum.Enum):
    """Enum for different model family types."""

    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    TOY_MODEL = "toy_model"


class ModelName(enum.Enum):
    """Enum for different model names.

    Model names are unique within their dataset-ModelType scope -- no two
    models of the same type trained on the same dataset can share a name.

    Model names are used for training, saving, and loading different model
    versions."""

    DEFAULT = "default"
