from train_model import load_data, split_data, train_model, evaluate_model
from sklearn.linear_model import LogisticRegression
import numpy as np


def test_load_data():
    X, y = load_data()
    assert X.shape[0] == y.shape[0], "Mismatch in number of samples between X and y"
    assert X.shape[1] == 4, "Expected 4 features in X"
    assert len(np.unique(y)) == 3, "Expected 3 unique classes in y"


def test_split_data():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    total_samples = X.shape[0]
    assert (
        len(X_train) + len(X_test) == total_samples
    ), "Total samples mismatch after splitting"
    assert (
        len(y_train) + len(y_test) == total_samples
    ), "Total labels mismatch after splitting"


def test_train_model():
    X, y = load_data()
    X_train, _, y_train, _ = split_data(X, y)
    model = train_model(X_train, y_train)
    assert isinstance(
        model, LogisticRegression
    ), "Model is not an instance of LogisticRegression"


def test_evaluate_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    score = evaluate_model(model, X_test, y_test)
    assert 0.0 <= score <= 1.0, "Model accuracy score is out of range"
