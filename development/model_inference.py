import mlflow
import mlflow.sklearn


mlflow.set_tracking_uri("sqlite:///mlflow.db")


MODEL_URI = "models:/CancerDetectionModel@production"


model = mlflow.sklearn.load_model(MODEL_URI)


print("Production model loaded successfully!")
print(model)