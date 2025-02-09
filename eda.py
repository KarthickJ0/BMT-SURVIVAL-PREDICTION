import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(input_file, output_dir):
    # Load the cleaned dataset
    data = pd.read_csv(input_file)

    # Create output directory for saving plots
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Basic Statistics
    print("Basic Statistics of the Dataset:")
    print(data.describe())

    # Step 2: Check for class imbalance in the target variable (assuming 'Survival' is the target column)
    if 'Survival' in data.columns:
        survival_counts = data['Survival'].value_counts()
        print("\nSurvival Class Distribution:")
        print(survival_counts)

        # Plot class distribution
        plt.figure(figsize=(6, 4))
        sns.barplot(x=survival_counts.index, y=survival_counts.values, palette='viridis')
        plt.title("Survival Class Distribution")
        plt.xlabel("Survival")
        plt.ylabel("Count")
        plt.savefig(os.path.join(output_dir, "survival_class_distribution.png"))
        plt.close()

    # Step 3: Correlation Heatmap
    plt.figure(figsize=(12, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close()

    # Step 4: Feature Distributions
    for column in data.columns:
        plt.figure(figsize=(6, 4))
        if data[column].dtype in ['float64', 'int64']:
            sns.histplot(data[column], kde=True, bins=30, color='blue')
            plt.title(f"Distribution of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.savefig(os.path.join(output_dir, f"distribution_{column}.png"))
            plt.close()

    print(f"EDA completed. Plots saved to {output_dir}")

if __name__ == "__main__":
    input_file = r"C:\Users\Arivumani\Desktop\BMT PROJECT\data\processed\cleaned_dataset.csv"
    output_dir = r"C:\Users\Arivumani\Desktop\BMT PROJECT\eda_plots"
    perform_eda(input_file, output_dir)
