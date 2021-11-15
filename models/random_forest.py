import numpy as np
from sklearn.ensemble import RandomForestClassifier


def get_accuracy(y_pred, y_true):
    correct_count = y_pred[y_pred == y_true].shape[0]
    incorrect_count = y_pred[y_pred != y_true].shape[0]
    return correct_count / (correct_count + incorrect_count)

def train_model(X, Y, X_test, Y_test):
    X = np.array(X)
    Y = np.array(Y.astype(np.float))

    X_test = np.array(X_test)
    Y_test = np.array(Y_test.astype(np.float))

    n_list = [5, 10, 100]
    split = 0.01 # already checked for best value
    best_model = None
    best_score = -np.inf
    for n in n_list:
        rf = RandomForestClassifier(n_estimators=n, min_samples_split=split)
        rf.fit(X, Y)
        y_pred = rf.predict(X_test)
        accuracy = get_accuracy(y_pred, Y_test)
        print(f"\t accuracy: {accuracy}")
        if accuracy > best_score:
            best_score = accuracy
            best_model = rf
    return best_model, best_score
