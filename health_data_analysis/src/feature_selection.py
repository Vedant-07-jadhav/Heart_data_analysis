import pandas as pd
from sklearn.model_selection import train_test_split

# Load the cleaned dataset
file_path = "../data/heart_cleaned.csv"
df = pd.read_csv(file_path)

# Define the target variable (assuming 'target' is the column name for heart disease)
target_column = "target"
X = df.drop(columns=[target_column])  # Features
y = df[target_column]  # Target

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the split datasets
X_train.to_csv("../data/X_train.csv", index=False)
X_test.to_csv("../data/X_test.csv", index=False)
y_train.to_csv("../data/y_train.csv", index=False)
y_test.to_csv("../data/y_test.csv", index=False)

print("\nFeature Selection & Data Splitting Completed!")
print(f"Training Set: {X_train.shape}, Test Set: {X_test.shape}")
