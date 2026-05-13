import unittest
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
import os
import pandas as pd

dagshub_token = os.environ("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com/"
repo_owner='email4prasanth'
repo_name='mlflow_ci'

mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
mlflow.set_experiment("Final_model")

model_name = "Best Model"
 
class TestModelLoading(unittest.TestCase):

    def test_model_in_staging(self):
        client = MlflowClient()
        versions = client.search_model_versions(
            f"name='{model_name}'"
        )
        self.assertGreater(len(versions),0, "No model found in Staing stage")

    def test_model_in_loading(self):
        client = MlflowClient()
        model_version = client.get_model_version_by_alias(
            model_name,
            "Staging"
        )

        version = model_version.version

        model_uri = f"models:/{model_name}@Staging"

        try:
            loaded_model = mlflow.pyfunc.load_model(model_uri)

        except Exception as e:
             self.fail(f"Failed to load {e}")

        self.assertIsNotNone(loaded_model, "The loaded model is None")
        print("model successfully loaded from {version}.")


if __name__ == "__main__":
     unittest.main()
