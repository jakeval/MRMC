import numpy as np
from sklearn.linear_model import SGDClassifier


def get_accuracy(y_pred, y_true):
    correct_count = y_pred[y_pred == y_true].shape[0]
    incorrect_count = y_pred[y_pred != y_true].shape[0]
    return correct_count / (correct_count + incorrect_count)


def train_model(X, Y, X_test, Y_test, class_weight=None):
    c_list = [1e-4, 1e-3, 1e-2, 1e1, 1e0, 1e1, 1e2, 1e3]
    best_model = None
    best_score = -np.inf
    for c in c_list:
        clf = SGDClassifier(loss='log', penalty='elasticnet', l1_ratio=0.15, class_weight=class_weight)
        clf.fit(X, Y)
        y_pred = clf.predict(X_test)
        accuracy = get_accuracy(y_pred, Y_test)
        print(f"\t c: {c}, accuracy: {accuracy}")
        if accuracy > best_score:
            best_score = accuracy
            best_model = clf
    return best_model, best_score
