import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

# Define file paths
ensemble_model_path = "C:/Users/Arivumani/Desktop/BMT PROJECT/ensemble_models/voting_classifier.pkl"
shap_output_dir = "C:/Users/Arivumani/Desktop/BMT PROJECT/shap_outputs"
test_file = "C:/Users/Arivumani/Desktop/BMT PROJECT/data/processed/cleaned_test_dataset.csv"

# Load the ensemble model
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)

# Load the test dataset
def load_test_data(test_file):
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found at {test_file}")
    return pd.read_csv(test_file)

# Generate SHAP plots
def generate_shap_plots(model, X_test, model_name):
    try:
        # Determine the appropriate SHAP explainer based on the model type
        if model_name == "LogisticRegression":
            masker = shap.maskers.Independent(X_test)
            explainer = shap.LinearExplainer(model, masker)
        elif model_name in ["RandomForest", "XGBoost"]:
            explainer = shap.TreeExplainer(model)
        else:
            # Use KernelExplainer for unsupported models
            explainer = shap.KernelExplainer(model.predict, shap.sample(X_test, 50))

        # Compute SHAP values
        shap_values = explainer(X_test)

        # Handle multiclass outputs for global feature importance
        if isinstance(shap_values, list) or shap_values.values.ndim == 3:  # Multiclass case
            shap_values_global = shap.Explanation(
                values=shap_values.values[:, :, 1] if shap_values.values.ndim == 3 else shap_values[1].values,
                base_values=shap_values.base_values[:, 1] if shap_values.values.ndim == 3 else shap_values[1].base_values,
                data=X_test
            )
        else:  # Binary or regression case
            shap_values_global = shap_values

        print(f"SHAP values shape for {model_name}: {shap_values_global.values.shape}")

        # Save global feature importance plot
        global_path = os.path.join(shap_output_dir, f"global_feature_importance_{model_name}.png")
        plt.figure()
        try:
            shap.summary_plot(shap_values_global, X_test, show=False)
            plt.savefig(global_path, bbox_inches='tight')
            print(f"Global feature importance plot saved to {global_path}")
        except Exception as e:
            print(f"Error generating global feature importance plot for {model_name}: {e}")
        finally:
            plt.close()

        # Save individual prediction explanation plot
        individual_path = os.path.join(shap_output_dir, f"individual_prediction_{model_name}.png")
        plt.figure()
        try:
            shap.waterfall_plot(shap.Explanation(
                values=shap_values[0].values if isinstance(shap_values, list) else shap_values.values[0],
                base_values=shap_values[0].base_values if isinstance(shap_values, list) else shap_values.base_values[0],
                data=X_test.iloc[0]
            ))
            plt.savefig(individual_path, bbox_inches='tight')
            print(f"Individual prediction explanation plot saved to {individual_path}")
        except Exception as e:
            print(f"Error generating individual prediction explanation plot for {model_name}: {e}")
        finally:
            plt.close()

    except ValueError as ve:
        print(f"ValueError generating SHAP plots for {model_name}: {ve}")
    except IndexError as ie:
        print(f"IndexError generating SHAP plots for {model_name}: {ie}")
    except Exception as e:
        print(f"Error generating SHAP plots for {model_name}: {e}")

# Main script
if __name__ == "__main__":
    print("Starting SHAP plots generation script...")

    # Load the model and test data
    ensemble_model = load_model(ensemble_model_path)
    test_data = load_test_data(test_file)

    # Ensure output directory exists
    os.makedirs(shap_output_dir, exist_ok=True)

    # Extract base models from the ensemble
    base_models = [(name, model) for name, model in ensemble_model.named_estimators_.items() if model is not None]

    for model_name, model in base_models:
        print(f"Generating SHAP plots for {model_name}...")

        try:
            # Ensure test data has the correct features
            expected_features = model.feature_names_in_
            X_test = test_data[expected_features]

            # Generate SHAP plots
            generate_shap_plots(model, X_test, model_name)

        except KeyError as ke:
            print(f"KeyError processing {model_name}: {ke}")
        except Exception as e:
            print(f"Error processing {model_name}: {e}")

    print("SHAP plots generation completed.")
