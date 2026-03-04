import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Set MLflow tracking URI to local directory
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Iris Classification Practice")

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Different hyperparameter combinations
param_combinations = [
    {"n_estimators": 10, "max_depth": 3, "random_state": 42},
    {"n_estimators": 50, "max_depth": 5, "random_state": 42},
    {"n_estimators": 100, "max_depth": 7, "random_state": 42},
]

for i, params in enumerate(param_combinations):
    with mlflow.start_run(run_name=f"Random Forest Run {i+1}"):
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log dataset info as artifact
        with open("dataset_info.txt", "w") as f:
            f.write(f"Iris Dataset\n")
            f.write(f"Features: {iris.feature_names}\n")
            f.write(f"Target classes: {iris.target_names}\n")
            f.write(f"Training samples: {len(X_train)}\n")
            f.write(f"Test samples: {len(X_test)}\n")
        
        mlflow.log_artifact("dataset_info.txt")
        
        print(f"Run {i+1}: Accuracy = {accuracy:.4f}")

print("\nMLflow runs completed! Open http://localhost:5000 to explore the UI.")