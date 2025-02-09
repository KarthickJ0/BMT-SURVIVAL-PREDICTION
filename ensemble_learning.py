import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def ensemble_learning(model_dir, test_file, ensemble_dir):
    # Load the test data
    data = pd.read_csv(test_file)
    
    # Convert the target variable to discrete classes if it's continuous
    if data["survival_status"].dtype in ["float64", "int64"]:
        # Example: Binarize survival_status (1 if survival_time > threshold, else 0)
        threshold = data["survival_status"].median()  # Use median as threshold
        data["survival_status"] = (data["survival_status"] > threshold).astype(int)
    
    # Separate features and target
    X_test = data.drop(columns=["survival_status"])  # Replace with the target column name
    y_test = data["survival_status"]

    # Load trained models
    model_files = [file for file in os.listdir(model_dir) if file.endswith(".pkl")]
    models = [(model_file.split(".")[0], joblib.load(os.path.join(model_dir, model_file))) for model_file in model_files]

    # Initialize Voting Classifier
    voting_clf = VotingClassifier(estimators=models, voting="soft")

    # Train the Voting Classifier on predictions (no training in real sense, just aggregation)
    voting_clf.fit(X_test, y_test)  # Fit step needed for compatibility

    # Get predictions
    y_pred = voting_clf.predict(X_test)
    y_proba = voting_clf.predict_proba(X_test)[:, 1] if hasattr(voting_clf, "predict_proba") else None

    # Evaluate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A"

    print(f"Ensemble VotingClassifier - Accuracy: {accuracy * 100:.2f}%, Precision: {precision * 100:.2f}%, Recall: {recall * 100:.2f}%, F1: {f1 * 100:.2f}%, AUC: {roc_auc * 100 if roc_auc != 'N/A' else roc_auc}%")

    # Save the VotingClassifier model
    os.makedirs(ensemble_dir, exist_ok=True)
    ensemble_path = os.path.join(ensemble_dir, "voting_classifier.pkl")
    joblib.dump(voting_clf, ensemble_path)
    print(f"Ensemble model saved to {ensemble_path}")

if __name__ == "__main__":
    model_dir = "C:/Users/Arivumani/Desktop/BMT PROJECT/models"
    test_file = "C:/Users/Arivumani/Desktop/BMT PROJECT/data/processed/cleaned_dataset.csv"
    ensemble_dir = "C:/Users/Arivumani/Desktop/BMT PROJECT/ensemble_models"
    ensemble_learning(model_dir, test_file, ensemble_dir)
