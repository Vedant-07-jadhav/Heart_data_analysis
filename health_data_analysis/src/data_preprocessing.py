import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("../data/heart.csv")

# Check class balance
print("Class Distribution:\n", df['target'].value_counts())

# Split features and target
X = df.drop(columns=['target'])
y = df['target']

# Identify categorical columns and encode them
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
if len(categorical_cols) > 0:
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)  # One-hot encoding

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Save balanced dataset
balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
balanced_df['target'] = y_resampled
balanced_df.to_csv("../data/heart_balanced.csv", index=False)

print("Balanced dataset saved!")
