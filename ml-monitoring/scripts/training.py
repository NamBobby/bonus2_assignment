import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def main():
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI", "http://localhost:25000"
    )
    os.environ.setdefault(
        "MLFLOW_S3_ENDPOINT_URL", "http://localhost:29000"
    )
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "minio")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minio123")

    model_name = os.getenv("MODEL_NAME", "wine_quality_model")
    output_dir = Path(__file__).resolve().parents[1] / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("wine_quality_experiment")

    dataset = load_wine()
    X = np.asarray(dataset.data, dtype=np.float64)
    y = dataset.target
    feature_names = list(dataset.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "random_state": 42,
    }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_score": float(f1_score(
            y_test, predictions, average="weighted", zero_division=0
        )),
        "precision": float(precision_score(
            y_test, predictions, average="weighted", zero_division=0
        )),
        "recall": float(recall_score(
            y_test, predictions, average="weighted", zero_division=0
        )),
    }

    reference = pd.DataFrame(X_train, columns=feature_names)
    reference_path = output_dir / "reference.csv"
    reference.to_csv(reference_path, index=False)

    schema = {
        "dataset": "sklearn.datasets.load_wine",
        "task": "three-class wine cultivar classification",
        "seed": 42,
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "class_names": list(dataset.target_names),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "mean": X_train.mean(axis=0).tolist(),
        "std": X_train.std(axis=0).tolist(),
    }
    schema_path = output_dir / "feature_schema.json"
    schema_path.write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )

    with mlflow.start_run(run_name="wine-random-forest") as run:
        mlflow.log_params(params)
        mlflow.log_param("feature_count", len(feature_names))
        mlflow.set_tags({
            "dataset": "sklearn-wine",
            "reference_source": "training_split",
            "task": "multiclass-classification",
        })
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(reference_path), "reference")
        mlflow.log_artifact(str(schema_path), "reference")

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name,
            signature=mlflow.models.infer_signature(
                X_train, model.predict(X_train)
            ),
            input_example=X_train[:5],
        )
        run_id = run.info.run_id

    client = MlflowClient()
    versions = client.search_model_versions(
        f"name = '{model_name}'"
    )
    matches = [v for v in versions if v.run_id == run_id]
    if len(matches) != 1:
        raise RuntimeError(
            "Cannot identify the model version for this training run."
        )

    version = matches[0].version
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )

    loaded = mlflow.pyfunc.load_model(
        f"models:/{model_name}/{version}"
    )
    reloaded_predictions = np.asarray(loaded.predict(X_test[:5]))
    np.testing.assert_array_equal(
        reloaded_predictions, model.predict(X_test[:5])
    )

    summary = {
        "model_name": model_name,
        "model_version": version,
        "run_id": run_id,
        "stage": "Production",
        "feature_count": len(feature_names),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "test_metrics": metrics,
        "registry_predictions_match": True,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print("REFERENCE:", reference_path)
    print("SCHEMA:", schema_path)
    print("TRAINING AND REGISTRY CHECK PASSED")


if __name__ == "__main__":
    main()
