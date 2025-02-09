import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def preprocess_data(input_file, output_file):
    # Load the dataset
    data = pd.read_csv(input_file)

    # Step 1: Replace placeholder missing values ('?') with NaN
    data.replace('?', pd.NA, inplace=True)

    # Step 2: Handle missing values using KNN Imputer
    # Separate categorical and numerical columns
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    # Impute numerical columns
    imputer = KNNImputer(n_neighbors=5)
    data[numerical_cols] = imputer.fit_transform(data[numerical_cols])

    # For categorical columns, fill missing values with the mode
    for col in categorical_cols:
        data[col].fillna(data[col].mode()[0], inplace=True)

    # Step 3: Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])
        label_encoders[col] = encoder

    # Step 4: Normalize numerical features
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # Step 5: Save the cleaned dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    data.to_csv(output_file, index=False)

    print(f"Data preprocessing completed. Cleaned data saved to {output_file}")

if __name__ == "__main__":
    input_file = r"C:\Users\Arivumani\Desktop\BMT PROJECT\dataset\bone-marrow-dataset.csv"  # Path to the raw dataset
    output_file = r"C:\Users\Arivumani\Desktop\BMT PROJECT\data\processed\cleaned_dataset.csv"  # Path to save the cleaned dataset
    preprocess_data(input_file, output_file)
