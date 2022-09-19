import numpy as np
from sklearn.linear_model import LogisticRegression


def get_accuracy(y_pred, y_true):
    correct_count = y_pred[y_pred == y_true].shape[0]
    incorrect_count = y_pred[y_pred != y_true].shape[0]
    return correct_count / (correct_count + incorrect_count)


def train_model(dataset, testset, preprocessor):
    dataset, testset = dataset.copy(), testset.copy()
    c_list = [1e-4, 1e-3, 1e-2, 1e1, 1e0, 1e1, 1e2]
    best_model = None
    best_score = -np.inf
    X = preprocessor.transform(dataset.drop('income', axis=1)).to_numpy()
    X_test = preprocessor.transform(testset.drop('income', axis=1)).to_numpy()
    
    Y = dataset['income'].to_numpy()
    Y[Y == '>50K'] = 1
    Y[Y != 1] = -1
    Y = Y.astype('int64')

    Y_test = testset['income'].to_numpy()
    Y_test[Y_test == '>50K'] = 1
    Y_test[Y_test != 1] = -1
    Y_test = Y_test.astype('int64')

    for c in c_list:
        lr = LogisticRegression(C=c, max_iter=10000)
        lr.fit(X, Y)
        y_pred = lr.predict(X_test)
        accuracy = get_accuracy(y_pred, Y_test)
        print(f"\t c: {c}, accuracy: {accuracy}")
        if accuracy > best_score:
            best_score = accuracy
            best_model = lr
    return best_model, best_score
