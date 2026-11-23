import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

def start_experiment(experiment_name):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)
    print(f"MLflow experiment started: {experiment_name}")

def start_run(run_name):
    mlflow.start_run(run_name=run_name)
    print(f"MLflow run started: {run_name}")

def log_parameters(params):
    mlflow.log_params(params)

def log_metrics(metrics):
    mlflow.log_metrics(metrics)

def log_tags(tags):
    mlflow.set_tags(tags)

def log_model(model, preprocessing_pipeline):
    production_pipeline = Pipeline([
        ("preprocessing", preprocessing_pipeline),
        ("model", model)
    ])
    mlflow.sklearn.log_model(
        sk_model=production_pipeline,
        artifact_path="model",
        serialization_format="pickle"
    )

def register_best_model(run_id, model_name="CancerDetectionModel"):
    client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
    model_uri = f"runs:/{run_id}/model"
    
    registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
    
    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=registered_model.version
    )
    print(f"✅ Successfully registered {model_name} (Version {registered_model.version}) with @production alias!")
