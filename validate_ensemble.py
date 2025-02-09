import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def validate_ensemble(ensemble_model_path, test_file):
    # Load the ensemble model
    if not os.path.exists(ensemble_model_path):
        raise FileNotFoundError(f"Ensemble model not found at {ensemble_model_path}")

    ensemble_model = joblib.load(ensemble_model_path)

    # Load the test dataset
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test dataset not found at {test_file}")

    test_data = pd.read_csv(test_file)

    # Ensure the target column is integer
    if test_data["survival_status"].dtype != np.int64:
        print("Converting survival_status to integer type.")
        test_data["survival_status"] = test_data["survival_status"].astype(int)

    # Separate features and target
    X_test = test_data.drop(columns=["survival_status"])
    y_test = test_data["survival_status"]

    print("\n=== Validating Ensemble Model ===")

    # Determine the number of classes in the target
    num_classes = len(np.unique(y_test))

    # Cross-validation on test data
    try:
        cross_val_scores = cross_val_score(ensemble_model, X_test, y_test, cv=5, scoring='accuracy')
        print(f"Cross-Validation Accuracy: {np.mean(cross_val_scores) * 100:.2f}%")
    except ValueError as e:
        print(f"Cross-validation failed: {e}")
        return

    # Evaluate on the unseen test dataset
    y_pred = ensemble_model.predict(X_test)

    # Multiclass AUC handling
    if num_classes > 2:
        y_proba = ensemble_model.predict_proba(X_test)
        auc = roc_auc_score(pd.get_dummies(y_test), y_proba, multi_class="ovr") * 100
    else:
        y_proba = ensemble_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba) * 100

    accuracy = accuracy_score(y_test, y_pred) * 100

    # Use appropriate averaging for precision, recall, and F1
    average_type = 'macro' if num_classes > 2 else 'binary'

    precision = precision_score(y_test, y_pred, average=average_type) * 100
    recall = recall_score(y_test, y_pred, average=average_type) * 100
    f1 = f1_score(y_test, y_pred, average=average_type) * 100

    print(f"Precision ({average_type}): {precision:.2f}%")
    print(f"Recall ({average_type}): {recall:.2f}%")
    print(f"F1 Score ({average_type}): {f1:.2f}%")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"AUC: {auc:.2f}%")

    # Robustness Check - Adding noise to test data
    print("\n=== Robustness Check ===")
    noise = np.random.normal(0, 0.01, X_test.shape)  # Small noise added to features
    X_test_noisy = X_test + noise
    y_pred_noisy = ensemble_model.predict(X_test_noisy)
    accuracy_noisy = accuracy_score(y_test, y_pred_noisy) * 100
    print(f"Accuracy on Noisy Data: {accuracy_noisy:.2f}%")

    if accuracy_noisy < accuracy - 5:
        print("Warning: The ensemble model's performance drops significantly with noise. Consider improving robustness.")
    else:
        print("The ensemble model is robust against small noise.")

if __name__ == "__main__":
    # File paths
    ensemble_model_path = "C:/Users/Arivumani/Desktop/BMT PROJECT/ensemble_models/voting_classifier.pkl"
    test_file = "C:/Users/Arivumani/Desktop/BMT PROJECT/data/processed/cleaned_test_dataset.csv"

    # Validate the ensemble model
    validate_ensemble(ensemble_model_path, test_file)
