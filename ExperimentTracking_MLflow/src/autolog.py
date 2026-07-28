# ---------------------------------------------------------#
#      This  file is demonstration for localhost mlflow  autologging  #
# ---------------------------------------------------------#

import mlflow
import mlflow.sklearn

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# --------------------------------------------------
# MLflow Setup
# --------------------------------------------------

project_dir = Path(__file__).resolve().parent.parent

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Wine_Classification")

print("Tracking URI:", mlflow.get_tracking_uri())


# --------------------------------------------------
#  AUTO LOGGING : 
# --------------------------------------------------


mlflow.autolog()


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

wine = load_wine()

X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=42
)

# --------------------------------------------------
# Hyperparameters
# --------------------------------------------------

max_depth = 100
n_estimators = 50

# --------------------------------------------------
# Start MLflow Run
# ---------------------------
# -----------------------

with mlflow.start_run():

    # # Log Parameters
    # mlflow.log_param("max_depth", max_depth)
    # mlflow.log_param("n_estimators", n_estimators)

    # Train Model
    model = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy :", accuracy)

    # Log Metric :

    # mlflow.log_metric("accuracy", accuracy)

    # Log Model :

    # mlflow.sklearn.log_model(
    #     sk_model=model,
    #     name="RandomForestModel"
    # )

    # Confusion Matrix 

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=wine.target_names,
        yticklabels=wine.target_names,
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    figure_path = project_dir / "Confusion_matrix.png"

    plt.savefig(figure_path)
    plt.close()


    # Log Artifacts :

    # mlflow.log_artifact(str(figure_path))
    # mlflow.log_artifact(__file__)

    # Log tags : 

    mlflow.set_tags({
        "Author" : "Ayush",
        "Project " : "Wine classification"
    })




print("\nExperiment Logged Successfully!")