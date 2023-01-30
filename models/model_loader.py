from data import data_loader
from models import model_interface
from models.core import logistic_regression
from models import model_constants


def load_model(
    model_type: model_constants.ModelType,
    dataset_name: data_loader.DatasetName,
    model_name: model_constants.ModelName = model_constants.ModelName.DEFAULT,
) -> model_interface.Model:
    """Loads the requested model from local disk.

    model_name is an identifier for the model unique within the model's family
    and training dataset.

    Args:
        model_type: The model family type to load.
        dataset_name: The name of the dataset the model was trained on.
        model_name: The name of the trained models.

    Returns:
        The requested model loaded from local disk."""
    if model_type == model_constants.ModelType.LOGISTIC_REGRESSION:
        lr = logistic_regression.LogisticRegression(dataset_name, model_name)
        return lr.load_model()
    elif model_type == model_constants.ModelType.TOY_MODEL:
        from confidence_checks import identity_adapter, toy_model, toy_data

        _, dataset_info = toy_data.get_data()
        adapter = identity_adapter.IdentityAdapter(
            label_column=dataset_info.label_column,
            positive_label=dataset_info.positive_label,
        )
        return toy_model.ToyModel(adapter)
    else:
        raise NotImplementedError(
            f"Model type {model_type} isn't implemented."
        )
