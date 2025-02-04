# **Heart Disease Prediction using Machine Learning**

## **Overview**
This project demonstrates a Flask-based web application that predicts the likelihood of heart disease based on user input using a trained machine learning model. It is built as part of a data science pipeline that includes preprocessing, model training, evaluation, and deployment.

The application uses a **Random Forest Classifier**, trained on balanced heart disease data, to predict outcomes. The model is accessible via a user-friendly web interface.

---

## **Features**
1. **Web Interface**:
   - Inputs patient data (e.g., age, blood pressure, cholesterol levels).
   - Predicts whether there is a risk of heart disease with results as either:
     - ✅ **No Heart Disease Detected**
     - ⚠️ **Heart Disease Detected — Consult a Doctor**

2. **Pipeline**:
   - **Data Preprocessing**: Handles imbalanced data using SMOTE and one-hot encodes categorical features.
   - **Model Training**: Implements a **Random Forest Classifier** with feature scaling.
   - **Model Evaluation**: Validates the model on testing data with accuracy and classification metrics.
   - **Deployed Model**: The trained model and scaler are saved and reused for predictions through Flask.

---

## **Technologies Used**
- **Machine Learning Libraries**: 
  - scikit-learn, imbalanced-learn, xgboost, pandas, numpy, joblib.
- **Web Framework**:
  - Flask (with Jinja2 templating for the frontend).
- **Visualization**:
  - matplotlib, seaborn.
- **Frontend**:
  - HTML, JavaScript (for form submission and API calls).

---

## **Folder Structure**
📁 health_data_analysis
│── 📁 data
│   ├── heart.csv
│   ├── heart_balanced.csv
│   ├── heart_cleaned.csv
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   ├── y_test.csv
│── 📁 models
│   ├── heart_disease_model.pkl  # Trained model
│   ├── scaler.pkl               # Scaler for preprocessing
│── 📁 notebooks
│── 📁 src
│   ├── app.py                   # Flask web application
│   ├── data_preprocessing.py     # Data cleaning & preparation
│   ├── feature_selection.py      # Feature selection methods
│   ├── main.py                   # Main execution script
│   ├── model_evaluation.py       # Model evaluation & metrics
│   ├── model_training.py         # Training the ML model
│   ├── trash.py                  # (Unused or test scripts)
│   ├── 📁 templates              # HTML frontend files
│── README.md                      # Project Documentation
│── requirements.txt               # Dependencies

## **Setup Instructions**

Follow the steps below to set up and run the project on your system:


### **1. Install dependencies**
Ensure you have Python (version 3.13+) installed. Then install required packages:
```bash
pip install -r requirements.txt
```

### **2. Train the model**
1. Preprocess the data using `data_preprocessing.py`:
   ```bash
   python scripts/data_preprocessing.py
   ```
2. Train the machine learning model using `model_training.py`:
   ```bash
   python scripts/model_training.py
   ```

### **3. Run the Flask application**
1. Start the Flask app to serve predictions using:
   ```bash
   python app/app.py
   ```
2. Open your browser and go to `http://127.0.0.1:5000/`.

### **4. Use the Web Interface**
- Input the required patient information into the form.
- Submit the form to get real-time predictions.

---

## **Dataset**
The dataset used is the **Heart Disease UCI dataset**. It contains medical information such as age, cholesterol, and max heart rate, along with a target variable indicating the presence of heart disease.

### Sample Data Features:
| Feature              | Description                                  |
|----------------------|----------------------------------------------|
| **Age**              | Age of the person                           |
| **Sex**              | Gender (1 = Male, 0 = Female)               |
| **CP**               | Chest pain type (0-3)                       |
| **Trestbps**         | Resting blood pressure                      |
| **Cholesterol**      | Serum Cholesterol (mg/dl)                   |
| **FBS**              | Fasting Blood Sugar (>120 mg/dl; 1 = True)  |
| **Thalach**          | Maximum heart rate                          |
| **Exang**            | Exercise-induced angina (1=Yes, 0=No)       |
| **Oldpeak**          | Depression induced after exercise           |
| **CA**               | Number of major vessels (0-3)               |
| **Thal**             | Thalassemia status (1-3)                    |
| **Target**           | Diagnosis (1 = Disease, 0 = Normal)         |

---

## **Model Evaluation Results**

Model performance was assessed using:
- **Accuracy**: The proportion of correct predictions.
- **Confusion Matrix**: Breakdown of True Positives, True Negatives, False Positives, and False Negatives.
- **Classification Report**: Precision, Recall, F1-score.

Example metrics (from training and evaluation):
- **Accuracy**: ~90%
- **Precision**: ~89%
- **Recall**: ~91%

---

## **API Endpoints**

### **1. `/` (GET)**
- Renders the HTML form for inputting patient features.

### **2. `/predict` (POST)**
- Accepts a JSON payload with `features` (13 individual patient features).
- Example request:
  ```json
  {
    "features": [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
  }
  ```
- Returns the prediction:
  ```json
  {
    "prediction": 1
  }
  ```

---

## **Key Features to Explore**
1. **Balanced Dataset**:
   - Addressed the class imbalance problem using SMOTE.
   - Improved model predictions for minority class detection.

2. **Web Deployment**:
   - Accessible user interface to make predictions.
   - Deployed via Flask for easy usage.

3. **Customizable Model**:
   - Easily modify the model or features to explore improvements.

---

## **Credits**
- **Dataset**: UCI Heart Disease Dataset.
- **Libraries**: scikit-learn, Flask, imbalanced-learn, joblib, etc.
- **Created By**: Vedant Jadhav

---

## **Contact**
If you have any questions, issues, or suggestions, feel free to reach out to:
- **Email**: `vedantdjadhav712@gmail.com`
