import unittest
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
import os
import pandas as pd

dagshub_token = os.getenv("DAGSHUB_TOKEN")

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
        print(f"model successfully loaded from version {version}.")

    def test_model_performance(self):
        client = MlflowClient()
        model_version = client.get_model_version_by_alias(
            model_name,
            "Staging"
        )

        version = model_version.version
        model_uri = f"models:/{model_name}@Staging"
        loaded_model = mlflow.pyfunc.load_model(model_uri)

        test_data_path = "./data/processed/test_processed.csv"
        if not os.path.exists(test_data_path):
            self.fail(f"Test data path not found {test_data_path}")
        test_data = pd.read_csv(test_data_path)
        x_test = test_data.drop(columns=['Potability'], axis=1)
        y_test = test_data['Potability']

        predicitons = loaded_model.predict(x_test)

        accuracy = accuracy_score(y_test , predicitons)
        precision =  precision_score(y_test, predicitons)
        recall = recall_score(y_test, predicitons)
        f1_s = f1_score(y_test, predicitons)

        print("accuracy",accuracy)
        print("precision",precision)
        print("recall", recall)
        print("f1_s", f1_s)

        self.assertGreaterEqual(accuracy, 0.7)
        self.assertGreaterEqual(precision, 0.3)
        self.assertGreaterEqual(recall, 0.3)
        self.assertGreaterEqual(f1_s, 0.3)

if __name__ == "__main__":
     unittest.main()
