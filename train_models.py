import numpy as np
from data import data_adapter as da
from models import regression, svm_clf, random_forest
from joblib import dump, load

def save_model(clf, file):
    dump(clf, file)

def load_german():
    train, test, preprocessor = da.load_german_credit_dataset()
    X = preprocessor.transform(train.drop('Y', axis=1)).to_numpy()
    Y = train.Y.to_numpy()
    X_test = preprocessor.transform(test.drop('Y', axis=1)).to_numpy()
    Y_test = test.Y.to_numpy()
    return X, Y, X_test, Y_test

def load_adult():
    train, test, preprocessor = da.load_adult_income_dataset()
    X = preprocessor.transform(train.drop('Y', axis=1)).to_numpy()
    Y = train.Y.to_numpy()
    X_test = preprocessor.transform(test.drop('Y', axis=1)).to_numpy()
    Y_test = test.Y.to_numpy()
    return X, Y, X_test, Y_test

def train_german_regression(X, Y, X_test, Y_test):
    model, accuracy = regression.train_model(X, Y, X_test, Y_test, class_weight='balanced')
    print("finished training")
    print(regression.get_accuracy(model.predict(X_test), Y_test))

def train_german_svc(X, Y, X_test, Y_test):
    model, accuracy = svm_clf.train_model(X, Y, X_test, Y_test, class_weight='balanced')
    print("finished training")
    print(svm_clf.get_accuracy(model.predict(X_test), Y_test))
    save_model(model, './saved_models/german_svc.pkl')

def train_german_rf(X, Y, X_test, Y_test):
    model, accuracy = random_forest.train_model(X, Y, X_test, Y_test, class_weight='balanced')
    print("finished training")
    print(svm_clf.get_accuracy(model.predict(X_test), Y_test))
    save_model(model, './saved_models/german_rf.pkl')

def train_adult_regression(X, Y, X_test, Y_test):
    model, accuracy = regression.train_model(X, Y, X_test, Y_test)
    print("finished training")
    print(regression.get_accuracy(model.predict(X_test), Y_test))

def train_adult_svc(X, Y, X_test, Y_test):
    model, accuracy = svm_clf.train_model(X, Y, X_test, Y_test)
    print("finished training")
    print(svm_clf.get_accuracy(model.predict(X_test), Y_test))
    save_model(model, './saved_models/adult_svc.pkl')

def train_adult_rf(X, Y, X_test, Y_test):
    model, accuracy = random_forest.train_model(X, Y, X_test, Y_test)
    print("finished training")
    print(svm_clf.get_accuracy(model.predict(X_test), Y_test))
    save_model(model, './saved_models/adult_rf.pkl')


if __name__ == '__main__':
    german_data = load_german()
    print("svc")
    train_german_svc(*german_data)
    print("random forest")
    train_german_rf(*german_data)


    adult_data = load_adult()
    print("svm")
    train_adult_svc(*adult_data)
    print("random forest")
    train_adult_rf(*adult_data)
    print("finished")
