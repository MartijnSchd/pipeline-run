import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature


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
        iris = load_iris()
        X = iris.data
        y = iris.target

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Train the model
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)

        # Evaluate the model
        score = model.score(X_test, y_test)
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

