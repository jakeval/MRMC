from joblib import load

def load_model(model, dataset):
    filename = None
    if dataset == 'german_credit':
        if model == 'svc':
            filename = './saved_models/german_svc.pkl'
        if model == 'random_forest':
            filename = './saved_models/german_rf.pkl'
    if dataset == 'adult_income':
        if model == 'svc':
            filename = './saved_models/adult_svc.pkl'
        if model == 'random_forest':
            filename = './saved_models/adult_rf.pkl'
    return load(filename)
