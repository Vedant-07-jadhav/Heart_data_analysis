import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load balanced dataset
df = pd.read_csv("../data/heart_balanced.csv")

# Split features & target
X = df.drop(columns=['target'])
y = df['target']

# Split into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Model
model = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42)
model.fit(X_train_scaled, y_train)

# Save model & scaler
joblib.dump(model, "../models/heart_disease_model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")

# Evaluate Model
y_pred = model.predict(X_test_scaled)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
