import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =====================
# MLflow Configuration
# =====================
mlflow.set_experiment("ci-telco-churn")

# =====================
# Argument Parser (CI)
# =====================
parser = argparse.ArgumentParser()
parser.add_argument("--max_iter", type=int, default=1000)
parser.add_argument("--C", type=float, default=1.0)
args = parser.parse_args()

# =====================
# Load Dataset
# =====================
data = pd.read_csv(
    r"telco-customer-churn_preprocessed.csv"
)

X = data.drop("Churn", axis=1)
y = data["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================
# Training + MLflow Run
# =====================
with mlflow.start_run() as run:

    model = LogisticRegression(
        max_iter=args.max_iter,
        C=args.C,
        solver="lbfgs"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # =====================
    # Metrics
    # =====================
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # =====================
    # MLflow Logging
    # =====================
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", args.max_iter)
    mlflow.log_param("C", args.C)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, artifact_path="model")

    # =====================
    # Output
    # =====================
    run_id = run.info.run_id

    print("===== CI Telco Churn Training =====")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"MLflow Run ID: {run_id}")

    # =====================
    # Send RUN_ID to GitHub Actions
    # =====================
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"RUN_ID={run_id}\n")
