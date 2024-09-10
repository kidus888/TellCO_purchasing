import mlflow

def setup_mlflow():
    mlflow.set_tracking_uri("http://localhost:5000")  # or your server location
    mlflow.set_experiment("Engagement Experience Analysis")
