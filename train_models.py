import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

def train_and_evaluate_models(input_file, model_dir):
    # Load the cleaned dataset
    data = pd.read_csv(input_file)

    # Convert the target variable to discrete classes if it is continuous
    if data["survival_status"].dtype == "float64" or data["survival_status"].dtype == "int64":
        # Example: Binarize survival_status (1 if survival_time > threshold, else 0)
        # Replace `threshold` with a suitable value for your dataset
        threshold = data["survival_status"].median()  # Example: using median as threshold
        data["survival_status"] = (data["survival_status"] > threshold).astype(int)

    # Separate features and target
    X = data.drop(columns=["survival_status"])  # Replace with your target column name
    y = data["survival_status"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True),  # SVM with probability for AUC
        "RandomForest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(eval_metric="logloss")  # Adjust for multiclass if needed
    }

    # Prepare directories
    os.makedirs(model_dir, exist_ok=True)

    # Train and evaluate each model
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # Metrics
        accuracy = accuracy_score(y_test, y_pred) * 100
        precision = precision_score(y_test, y_pred) * 100
        recall = recall_score(y_test, y_pred) * 100
        f1 = f1_score(y_test, y_pred) * 100
        roc_auc = (roc_auc_score(y_test, y_proba) * 100) if y_proba is not None else "N/A"

        print(f"{name} - Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1: {f1:.2f}%, AUC: {roc_auc if roc_auc == 'N/A' else f'{roc_auc:.2f}%'}")

        # Save the trained model
        model_path = os.path.join(model_dir, f"{name}.pkl")
        joblib.dump(model, model_path)
        print(f"{name} saved to {model_path}")

if __name__ == "__main__":
    input_file = "C:/Users/Arivumani/Desktop/BMT PROJECT/data/processed/cleaned_dataset.csv"
    model_dir = "C:/Users/Arivumani/Desktop/BMT PROJECT/models"
    train_and_evaluate_models(input_file, model_dir)
