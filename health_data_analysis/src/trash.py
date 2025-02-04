from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Define paths to model and scaler
base_dir = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(base_dir, "../models/heart_disease_model.pkl")
scaler_path = os.path.join(base_dir, "../models/scaler.pkl")

# Verify that the model and scaler files exist
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file was not found at {model_path}")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"The scaler file was not found at {scaler_path}")

# Load the trained model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)


# Route to make predictions


        # Scale the features
        scaled_features = scaler.transform(features)

        # Make prediction
        prediction = model.predict(scaled_features)

        # Return the prediction as JSON
        result = {'prediction': int(prediction[0])}
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
