# Databricks notebook source
import mlflow
import pandas as pd

# Load the model
model_name = "IrisClassifier"
model_uri = f"models:/{model_name}@hi"  # Correct alias syntax
model = mlflow.sklearn.load_model(model_uri)


# Prepare input data
data = {
    'sepal length (cm)': [5.1, 6.2],
    'sepal width (cm)': [3.5, 2.8],
    'petal length (cm)': [1.4, 4.8],
    'petal width (cm)': [0.2, 1.8]
}
input_df = pd.DataFrame(data)

# Perform inference
predictions = model.predict(input_df)
print("Predictions:", predictions)

