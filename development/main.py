import os
import sys
import mlflow

# Ensure imports work regardless of which directory main.py is run from
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(CURRENT_DIR)

from data_load import load_data
from feature_engineering import create_features
from feature_selection import select_features
from train_test_split import split_data
from data_preprocessing import create_preprocessing_pipeline
from model_training import train_models
from model_evaluation import evaluate_models
from handle_imbalance import handle_imbalance
from mlflow_tracking import (
    start_experiment, start_run, log_parameters, 
    log_metrics, log_tags, log_model, register_best_model
)

# Start MLflow experiment
start_experiment("Cancer Detection")

# 1. Load Data
# 1. Load Data directly from AWS S3 Bucket (Industry Standard!)
file_path = "s3://cancer-detection-ml-project-bucket/cancer_detection_dataset.csv"
df = load_data(file_path)

# 2. Feature Engineering
df = create_features(df)

# 3. Feature Selection
X, y = select_features(df)

# 4. Train Test Split
X_train, X_test, y_train, y_test = split_data(X, y)

# 5. Create Preprocessing Pipeline
preprocessing_pipeline = create_preprocessing_pipeline()

# 6. Fit preprocessing only on training data
X_train_processed = preprocessing_pipeline.fit_transform(X_train)
X_test_processed = preprocessing_pipeline.transform(X_test)

# Track the best model based on Recall (Crucial for Cancer Detection!)
best_recall = -1.0
best_run_id = None

# 7. Handle Class Imbalance
methods = ["smote", "random_over", "random_under"]

for method in methods:
    X_train_resampled, y_train_resampled = handle_imbalance(
        X_train_processed, y_train, method=method
    )

    print(f"\nClass Distribution Before Resampling:")
    print(y_train.value_counts())

    print(f"\nClass Distribution After {method.upper()}:")
    print(y_train_resampled.value_counts())

    # 8. Train Multiple Models
    trained_models = train_models(X_train_resampled, y_train_resampled)

    # 9. Evaluate Multiple Models
    results_df = evaluate_models(trained_models, X_test_processed, y_test)
    print(f"\nModel Evaluation Results ({method}):")
    print(results_df)

    # 10. Start MLflow Run for Each Model
    for model_name, model in trained_models.items():
        start_run(f"{method} - {model_name}")
        
        current_run_id = mlflow.active_run().info.run_id

        log_parameters(model.get_params())

        metrics_row = results_df[results_df["Model"] == model_name].iloc[0]
        recall_val = float(metrics_row["Recall"])
        
        metrics = {
            "accuracy": float(metrics_row["Accuracy"]),
            "precision": float(metrics_row["Precision"]),
            "recall": recall_val,
            "f1_score": float(metrics_row["F1 Score"])
        }
        print(f"\nLogging metrics for {model_name}: {metrics}")
        log_metrics(metrics)

        # Tags
        tags = {
            "model_name": model_name,
            "imbalance_method": method,
            "experiment_type": "imbalance_comparison"
        }
        log_tags(tags)

        # Model Artifact
        log_model(model, preprocessing_pipeline)

        # Check if this model has the highest Recall so far
        if recall_val > best_recall:
            best_recall = recall_val
            best_run_id = current_run_id

        mlflow.end_run()

# Register the highest Recall model to production automatically!
if best_run_id:
    print(f"\n🏆 Best Model Recall: {best_recall:.4f} (Run ID: {best_run_id})")
    register_best_model(best_run_id, model_name="CancerDetectionModel")
