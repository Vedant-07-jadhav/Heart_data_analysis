import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load preprocessed data & model
data = np.load("../models/processed_data.npz")
X_test, y_test = data['X_test'], data['y_test']
model = joblib.load("../models/heart_disease_model.pkl")

# Make predictions
y_pred = model.predict(X_test)

# Print Evaluation Metrics
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
