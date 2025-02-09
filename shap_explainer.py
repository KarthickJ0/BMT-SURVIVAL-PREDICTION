import os
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def explain_voting_classifier(ensemble_model_path, test_file, shap_output_dir):
    # Ensure output directory exists
    os.makedirs(shap_output_dir, exist_ok=True)
    
    # Load the ensemble model
    if not os.path.exists(ensemble_model_path):
        raise FileNotFoundError(f"Ensemble model not found at {ensemble_model_path}")
    print("Loading ensemble model...")
    ensemble_model = joblib.load(ensemble_model_path)
    
    # Load the test dataset
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test dataset not found at {test_file}")
    print("Loading test dataset...")
    test_data = pd.read_csv(test_file)

    # Separate features and target
    X_test = test_data.drop(columns=["survival_status"])
    print("\n=== Generating SHAP Explanations for Voting Classifier ===")
    
    # Iterate over individual models in the VotingClassifier
    for name, model in ensemble_model.estimators:
        print(f"Processing model: {name}")
        
        try:
            # Determine the appropriate SHAP explainer
            if name == "LogisticRegression":
                explainer = shap.LinearExplainer(model, X_test)
                shap_values = explainer.shap_values(X_test)
            elif name == "SVM":
                explainer = shap.KernelExplainer(model.predict_proba, X_test)
                shap_values = explainer.shap_values(X_test)
            elif name in ["RandomForest", "XGBoost"]:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
            else:
                print(f"Model type not supported for SHAP: {name}. Skipping...")
                continue

            # Generate global feature importance plot
            global_feature_path = os.path.join(shap_output_dir, f"global_feature_importance_{name}.png")
            shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
            plt.title(f"Global Feature Importance - {name}")
            plt.savefig(global_feature_path, bbox_inches='tight')
            print(f"Global feature importance plot for {name} saved at {global_feature_path}")
            plt.close()

            # Generate individual prediction explanation for the first instance
            individual_feature_path = os.path.join(shap_output_dir, f"individual_prediction_{name}.png")
            instance = X_test.iloc[0]

            # Handle scalar vs array expected_value
            expected_value = (
                explainer.expected_value[0]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else explainer.expected_value
            )

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            # Use matplotlib for force plot saving for LogisticRegression and XGBoost
            if name in ["LogisticRegression", "XGBoost"]:
                shap.initjs()  # Initialize JS rendering (if needed)

                force_plot = shap.force_plot(
                    expected_value,
                    shap_values[0],
                    instance,
                    matplotlib=True,  # Use matplotlib for rendering
                    show=False
                )

                # Save the force plot as an image
                plt.savefig(individual_feature_path, bbox_inches='tight')
                print(f"Individual prediction explanation for {name} saved at {individual_feature_path}")
                plt.close()
            
            # For RandomForest and SVM, limit to a single sample
            if name in ["RandomForest", "SVM"]:
                shap.initjs()  # Initialize JS rendering (if needed)
                
                force_plot = shap.force_plot(
                    expected_value,
                    shap_values[0],
                    instance,
                    matplotlib=False,  # Set to False for rendering in notebook
                    show=False
                )

                # Save the force plot as an interactive HTML file
                interactive_force_path = os.path.join(shap_output_dir, f"individual_prediction_{name}_interactive.html")
                shap.save_html(interactive_force_path, force_plot)
                print(f"Interactive individual prediction explanation for {name} saved at {interactive_force_path}")

        except Exception as e:
            print(f"Skipping model '{name}' due to error: {e}")
    
    print("\nSHAP explanations generated for all supported models in the VotingClassifier.")


if __name__ == "__main__":
    # File paths
    ensemble_model_path = "C:/Users/Arivumani/Desktop/BMT PROJECT/ensemble_models/voting_classifier.pkl"
    test_file = "C:/Users/Arivumani/Desktop/BMT PROJECT/data/processed/cleaned_test_dataset.csv"
    shap_output_dir = "C:/Users/Arivumani/Desktop/BMT PROJECT/shap_outputs"

    print("Starting SHAP explainer script...")
    explain_voting_classifier(ensemble_model_path, test_file, shap_output_dir)
    print("SHAP explainer script completed.")
