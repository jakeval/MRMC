from joblib import load
import pathlib


RELATIVE_MODEL_DIR = 'saved_models'
ABSOLUTE_MODEL_DIR = pathlib.Path(__file__).parent.parent / RELATIVE_MODEL_DIR


def load_model(model, dataset):
    filename = None
    if dataset == 'german_credit':
        if model == 'svc':
            filename = ABSOLUTE_MODEL_DIR / 'german_svc.pkl'
        if model == 'random_forest':
            filename = ABSOLUTE_MODEL_DIR / 'german_rf.pkl'
    if dataset == 'adult_income':
        if model == 'svc':
            filename = ABSOLUTE_MODEL_DIR / 'adult_svc.pkl'
        if model == 'random_forest':
            filename = ABSOLUTE_MODEL_DIR / 'adult_rf.pkl'
    return load(filename)
