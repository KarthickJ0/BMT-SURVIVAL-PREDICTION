import os
import joblib
import shap
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# Define file paths
ensemble_model_path = "C:/Users/Arivumani/Desktop/BMT PROJECT/ensemble_models/voting_classifier.pkl"
shap_output_dir = "C:/Users/Arivumani/Desktop/BMT PROJECT/shap_outputs"
test_file = "C:/Users/Arivumani/Desktop/BMT PROJECT/data/processed/cleaned_test_dataset.csv"

# Load the ensemble model
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

# Load the test dataset
@st.cache_data
def load_test_data(test_file):
    return pd.read_csv(test_file)

# Display SHAP images with validation
def display_shap_images(model_name):
    global_path = os.path.join(shap_output_dir, f"global_feature_importance_{model_name}.png")
    individual_path = os.path.join(shap_output_dir, f"individual_prediction_{model_name}.png")

    # Validate Global Feature Importance Plot
    if os.path.exists(global_path) and os.path.getsize(global_path) > 0:
        st.image(global_path, caption=f"Global Feature Importance for {model_name}", use_container_width=True)
    else:
        st.warning(f"Global feature importance plot not found or invalid for {model_name}.")

    # Validate Individual Prediction Explanation Plot
    if os.path.exists(individual_path) and os.path.getsize(individual_path) > 0:
        st.image(individual_path, caption=f"Individual Prediction Explanation for {model_name}", use_container_width=True)
    else:
        st.warning(f"Individual prediction explanation not found or invalid for {model_name}.")

# Real-time SHAP explanation
def explain_prediction(model, feature_values, model_name):
    try:
        feature_array = np.array([feature_values])  # Convert feature values to array
        shap_values_array = None
        if model_name == "LogisticRegression":
            # Use a masker for LinearExplainer
            masker = shap.maskers.Independent(feature_array)
            explainer = shap.LinearExplainer(model, masker)
            shap_values = explainer(feature_array)
        elif model_name in ["RandomForest", "XGBoost"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(feature_array)
        else:
            st.error(f"SHAP explainer not supported for {model_name}.")
            return

        if shap_values.values is not None and shap_values.values.size > 0:
            # Display SHAP explanation
            st.subheader(f"SHAP Explanation for {model_name}")
            st.write("Feature contributions to the prediction:")

            # Create a DataFrame for SHAP values
            shap_values_df = pd.DataFrame({
                "Feature": feature_values.index,
                "Contribution": shap_values.values[0]
            }).set_index("Feature")

            # Display feature contributions as a bar chart
            st.bar_chart(shap_values_df["Contribution"])
            
            # Optionally display as a table
            st.write("Feature Contributions Table:")
            st.dataframe(shap_values_df)
        else:
            st.error("SHAP values could not be generated or are invalid.")
    except Exception as e:
        st.error(f"Error during SHAP explanation: {e}")

# Predict survival and relapse
def predict_outcome(model, feature_values):
    try:
        feature_array = np.array([feature_values])
        prediction = model.predict(feature_array)
        prediction_prob = model.predict_proba(feature_array)

        survival_prob = prediction_prob[0][1] * 100  # Probability of survival (assuming binary classification)
        relapse_prob = prediction_prob[0][0] * 100   # Probability of relapse (or non-survival)

        st.subheader("Prediction Results")
        st.write(f"**Survival Probability:** {survival_prob:.2f}%")
        st.write(f"**Relapse Probability:** {relapse_prob:.2f}%")

        if prediction[0] == 1:
            st.success("The model predicts that the patient is likely to survive.")
        else:
            st.error("The model predicts that the patient is unlikely to survive (potential relapse).")
    except Exception as e:
        st.error(f"Error during prediction: {e}")


# Set constraints for each feature
def validate_feature_input(feature_name, default_value, min_value, max_value):
    # Clamp the default value to ensure it is within the specified range
    adjusted_default_value = max(float(min_value), min(float(default_value), float(max_value)))
    return st.sidebar.number_input(
        f"{feature_name}",
        value=adjusted_default_value,  # Use adjusted default value
        min_value=float(min_value),   # Convert min_value to float
        max_value=float(max_value),   # Convert max_value to float
        step=0.01                      # Keep step as a float
    )


# Streamlit App
st.title("SHAP Visualizations for Ensemble Model")

# Load the ensemble model and test dataset
ensemble_model = load_model(ensemble_model_path)
test_data = load_test_data(test_file)

# Extract base models from the ensemble
base_models = [(name, model) for name, model in ensemble_model.named_estimators_.items() if model is not None]

# Sidebar for model selection
st.sidebar.title("Model Selection")
selected_model_name = st.sidebar.selectbox("Select a model to analyze:", [name for name, _ in base_models])

# Display SHAP visualizations for the selected model
st.header(f"SHAP Visualizations for {selected_model_name}")
display_shap_images(selected_model_name)

# Input section for real-time prediction explanation
st.sidebar.title("Real-Time Prediction Explanation")
st.sidebar.write("Provide feature values for prediction:")
feature_values = {}

# Define constraints for each feature (example limits provided)
feature_constraints = {
    "donor_age": (18, 65),
    "donor_age_below_35": (0, 1),
    "donor_ABO": (0, 1),
    "donor_CMV": (0, 1),
    "recipient_age": (0, 100),
    "recipient_age_below_10": (0, 1),
    "recipient_age_int": (0, 1),
    "recipient_gender": (0, 1),
    "recipient_body_mass": (10, 150),
    "recipient_ABO": (0, 1),
    "recipient_rh": (0, 1),
    "recipient_CMV": (0, 1),
    "disease": (0, 5),
    "disease_group": (0, 5),
    "gender_match": (0, 1),
    "ABO_match": (0, 1),
    "CMV_status": (0, 1),
    "HLA_match": (0, 10),
    "HLA_mismatch": (0, 10),
    "antigen": (0, 10),
    "allel": (0, 10),
    "HLA_group_1": (0, 10),
    "risk_group": (0, 5),
    "stem_cell_source": (0, 1),
    "tx_post_relapse": (0, 1),
    "CD34_x1e6_per_kg": (0, 100),
    "CD3_x1e8_per_kg": (0, 500),
    "CD3_to_CD34_ratio": (0, 1000),
    "ANC_recovery": (0, 1),
    "time_to_ANC_recovery": (0, 100),
    "PLT_recovery": (0, 1),
    "time_to_PLT_recovery": (0, 100),
    "acute_GvHD_II_III_IV": (0, 1),
    "acute_GvHD_III_IV": (0, 1),
    "time_to_acute_GvHD_III_IV": (0, 100),
    "extensive_chronic_GvHD": (0, 1),
    "relapse": (0, 1),
    "survival_time": (0, 100)
}

for feature, (min_val, max_val) in feature_constraints.items():
    feature_values[feature] = validate_feature_input(feature, float(test_data[feature].mean()), min_val, max_val)

if st.sidebar.button("Predict and Explain"):
    selected_model = dict(base_models)[selected_model_name]
    predict_outcome(selected_model, pd.Series(feature_values))
    explain_prediction(selected_model, pd.Series(feature_values), selected_model_name)

st.write("\n\n---")
st.write("Developed for the Leukemia Survival Prediction Project.")
