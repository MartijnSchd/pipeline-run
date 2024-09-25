import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature


def load_data():
    """Load the Iris dataset."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

def split_data(X, y, test_size=0.2):
    """Split the dataset into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=42)

def train_model(X_train, y_train, max_iter=200):
    """Train the Logistic Regression model."""
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's accuracy."""
    score = model.score(X_test, y_test)
    return score


def main():
    experiment_name = "/Shared/IrisExperiment"  # You can choose any valid path
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        # Set a description for the run
        mlflow.set_tag("description", "Logistic Regression model for the Iris dataset")

        # Set custom tags
        mlflow.set_tag("model_type", "Logistic Regression")
        mlflow.set_tag("dataset", "Iris")
        # Load the data
        X, y = load_data()
        X_train, X_test, y_train, y_test = split_data(X, y)

        # Train the model
        model = train_model(X_train, y_train)

        # Evaluate the model
        score = evaluate_model(model, X_test, y_test)
        print(f"Model accuracy: {score}")

        # Log parameters and metrics
        mlflow.log_param('max_iter', 200)
        mlflow.log_metric('accuracy', score)

        #  Log and register the model with input example and signature
        input_example = X_test[:5]  # A few example inputs
        signature = infer_signature(X_test, model.predict(X_test))
        
        mlflow.sklearn.log_model(model, 'model', signature=signature, input_example=input_example)
        mlflow.register_model(f'runs:/{mlflow.active_run().info.run_id}/model', 'IrisClassifier')

if __name__ == "__main__":
    main()

