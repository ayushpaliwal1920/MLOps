
# ---------------------------------------------------------------------------#
#      This  file is demonstration for dagshub/team / remote host  mlflow    #
# ---------------------------------------------------------------------------#


import platform
from pathlib import Path

import dagshub
import mlflow
import mlflow.sklearn

import matplotlib.pyplot as plt
import seaborn as sns

from mlflow.models import infer_signature

from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


# ==========================================================
# Initialize DagsHub + MLflow
# ==========================================================

dagshub.init(
    repo_owner="ayushpaliwal1920",
    repo_name="MLOps",
    mlflow=True
)

# DO NOT call mlflow.set_tracking_uri() again

mlflow.set_experiment("Wine_Classification")

print("Tracking URI :", mlflow.get_tracking_uri())


# ==========================================================
# Project Directory
# ==========================================================

project_dir = Path(__file__).resolve().parent.parent


# ==========================================================
# Load Dataset
# ==========================================================

wine = load_wine()

X = wine.data
y = wine.target


# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=42
)


# ==========================================================
# Hyperparameters
# ==========================================================

max_depth = 100
n_estimators = 50


# ==========================================================
# Start MLflow Run
# ==========================================================

with mlflow.start_run(run_name="RandomForest_Wine"):

    # ----------------------------
    # Parameters
    # ----------------------------

    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size", 0.1)
    mlflow.log_param("dataset", "Wine")
    mlflow.log_param("n_features", X.shape[1])
    mlflow.log_param("n_classes", len(wine.target_names))
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("test_samples", len(X_test))

    # ----------------------------
    # Train Model
    # ----------------------------

    model = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=42
    )

    model.fit(X_train, y_train)

    # ----------------------------
    # Prediction
    # ----------------------------

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy : {accuracy:.4f}")

    # ----------------------------
    # Metrics
    # ----------------------------

    mlflow.log_metric("accuracy", accuracy)

    # ----------------------------
    # Model Signature
    # ----------------------------

    signature = infer_signature(
        X_train,
        model.predict(X_train)
    )

    # ----------------------------
    # Log Model
    # ----------------------------

    mlflow.sklearn.log_model(
        sk_model=model,
        name="RandomForestModel",
        signature=signature
    )

    # ----------------------------
    # Confusion Matrix
    # ----------------------------

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=wine.target_names,
        yticklabels=wine.target_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    figure_path = project_dir / "Confusion_matrix.png"

    plt.savefig(figure_path)

    plt.close()

    # ----------------------------
    # Log Artifacts
    # ----------------------------

    mlflow.log_artifact(str(figure_path))

    mlflow.log_artifact(str(Path(__file__)))

    # ----------------------------
    # Feature Names
    # ----------------------------

    mlflow.log_dict(
        {"feature_names": wine.feature_names},
        "feature_names.json"
    )

    # ----------------------------
    # Target Names
    # ----------------------------

    mlflow.log_dict(
        {"classes": wine.target_names.tolist()},
        "target_names.json"
    )

    # ----------------------------
    # Tags
    # ----------------------------

    mlflow.set_tags({
        "author": "Ayush Paliwal",
        "project": "Wine Classification",
        "framework": "Scikit-Learn",
        "algorithm": "Random Forest",
        "python_version": platform.python_version(),
        "sklearn_version": sklearn_version,
        "tracking": "DagsHub"
    })

print("\nExperiment Logged Successfully!")