echo off
setx MLFLOW_TRACKING_URI sqlite:///mlflow.sqlite.db
mlflow server --host 0.0.0.0 --backend-store-uri %MLFLOW_TRACKING_URI% --default-artifact-root models