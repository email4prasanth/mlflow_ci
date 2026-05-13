# model_evaluation.py (corrected version)

import pandas as pd
import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
import numpy as np

import mlflow
import dagshub
import seaborn as sns
import matplotlib.pyplot as plt
from mlflow import log_metric, log_param, log_artifact
import mlflow.sklearn
from mlflow.models import infer_signature

dagshub.init(repo_owner='email4prasanth', repo_name='mlflow_ci', mlflow=True)
mlflow.set_experiment("DVC-Pipeline")

def load_model(filepath: str):
    try:
        with open(filepath, "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f"Error loading model from {filepath}: {e}")

def evaluate_model(model, y_test, y_pred, model_name: str) -> dict:
    try:
        params = yaml.safe_load(open("params.yaml", "r"))
        test_size = params["data_collection"]["test_size"]
        n_estimators = params["model_building"]["n_estimators"]

        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1s = f1_score(y_test, y_pred)

        mlflow.log_param("Test_size", test_size)
        mlflow.log_param("n_estimators", n_estimators) 

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", pre)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1s)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix for {model_name}")
        cm_path = f"confusion_matrix_{model_name.replace(' ', '_')}.png"
        plt.savefig(cm_path)
        plt.close()  # Close to free memory

        # Log confusion matrix artifact
        mlflow.log_artifact(cm_path)

        metrics_dict = {
            'accuracy': acc,
            'precision': pre,
            'recall': rec,
            'f1_score': f1s
        }
        return metrics_dict
    except Exception as e:
        raise Exception(f"Error in evaluating model: {e}")

def save_metrics(metrics: dict, metrics_path: str) -> None:
    try: 
        with open(metrics_path, 'w') as file:
            json.dump(metrics, file, indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {metrics_path}: {e}")

def main():
    # y_test and y_pred
    test_data_path = r"./data/processed/test_processed.csv"
    model_path = "models/model.pkl"
    metrics_path = "reports/metrics.json"
    model_name = "Best Model"
    
    # Load test data
    test_df = pd.read_csv(test_data_path)
    x_test = test_df.drop(columns=['Potability'], axis=1)
    y_test = test_df['Potability']
    
    # Load model
    model = load_model(model_path)
    
    # Make predictions
    y_pred = model.predict(x_test)
    
    # MLflow starts
    with mlflow.start_run() as run:
        try:
            # Evaluate model
            metrics = evaluate_model(model, y_test, y_pred, model_name)
            save_metrics(metrics, metrics_path)
            
            # Create signature for the model
            signature = infer_signature(x_test, y_pred)
            
            # Log the model in MLflow format
            # Make sure artifact_path is simple and doesn't have spaces
            artifact_path = "Best_Model"
            
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                signature=signature,
                registered_model_name=model_name
            )
            
            # Log additional artifacts
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(metrics_path)
            mlflow.log_artifact(__file__)
            
            # Verify the model was logged correctly
            # Check if the artifact directory exists
            client = mlflow.tracking.MlflowClient()
            artifacts = client.list_artifacts(run.info.run_id)
            print(f"\n📁 Artifacts logged in this run:")
            for artifact in artifacts:
                print(f"  - {artifact.path} (is_dir: {artifact.is_dir})")
            
            # Check inside Best_Model directory
            if any(a.path == artifact_path for a in artifacts):
                model_artifacts = client.list_artifacts(run.info.run_id, path=artifact_path)
                print(f"\n📦 Contents of {artifact_path}/:")
                for ma in model_artifacts:
                    print(f"  - {ma.path}")
            
            # Save run ID and model info to JSON File
            run_info = {
                'run_id': run.info.run_id, 
                'model_name': model_name,
                'artifact_path': artifact_path
            }
            reports_path = "reports/run_info.json"
            with open(reports_path, 'w') as file:
                json.dump(run_info, file, indent=4)
            
            print(f"\n✅ Model logged successfully with run_id: {run.info.run_id}")
            print(f"✅ Model URI: runs:/{run.info.run_id}/{artifact_path}")
            
        except Exception as e:
            print(f"❌ Error in main: {e}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == "__main__":
    main()