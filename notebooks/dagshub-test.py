import dagshub
dagshub.init(repo_owner='email4prasanth', repo_name='mlflow_ci', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)