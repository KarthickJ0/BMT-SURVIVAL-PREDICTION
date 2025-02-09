import pandas as pd
from sklearn.model_selection import train_test_split
import os

def create_test_dataset(input_file, output_dir, test_filename):
    """
    Create a test dataset from the processed dataset and save it to a specified location.

    Parameters:
    input_file (str): Path to the input processed dataset (cleaned dataset).
    output_dir (str): Directory to save the test dataset.
    test_filename (str): Filename for the test dataset.
    """
    # Load the processed dataset
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    data = pd.read_csv(input_file)

    # Split into training and test sets
    _, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the test dataset
    test_file_path = os.path.join(output_dir, test_filename)
    test_data.to_csv(test_file_path, index=False)

    print(f"Test dataset created and saved to {test_file_path}")

if __name__ == "__main__":
    # File paths
    input_file = "C:/Users/Arivumani/Desktop/BMT PROJECT/data/processed/cleaned_dataset.csv"
    output_dir = "C:/Users/Arivumani/Desktop/BMT PROJECT/data/processed"
    test_filename = "cleaned_test_dataset.csv"

    # Generate the test dataset
    create_test_dataset(input_file, output_dir, test_filename)
