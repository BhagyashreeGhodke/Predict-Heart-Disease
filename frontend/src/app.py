import os
import pickle
import gdown
import pandas as pd
from flask import Flask, render_template, request
import numpy as np
import traceback

# Initialize the Flask application
app = Flask(__name__)

# --- Model Downloading and Loading ---

# Define Google Drive URLs for all three pre-trained models
MODEL_URLS = {
    "logistic_regression_model.pkl": "https://drive.google.com/uc?id=1yZLf84PhhNhCa1Y6TfRf0io8RN9t__LE",
    "gradient_boosting_model.pkl": "https://drive.google.com/uc?id=1zGm5yaT_gykP2aqBw6KybB1l78MtvUsb",
    "random_forest_model.pkl": "https://drive.google.com/uc?id=176I6UzHhq0gcx2iGlXeRpyFEE1e3p8BW"

}

# Set up a local path to store the models
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_PATH, exist_ok=True)

# Download the models from Google Drive if they don't exist locally
for model_name, url in MODEL_URLS.items():
    file_path = os.path.join(MODEL_PATH, model_name)
    if not os.path.exists(file_path):
        print(f"Downloading {model_name}...")
        try:
            gdown.download(url, file_path, quiet=False)
            print(f"Successfully saved {model_name} to {file_path}")
        except Exception as e:
            print(f"Error downloading {model_name}: {e}")
    else:
        print(f"{model_name} already exists locally.")

# Load the downloaded models into memory
lr_model, gb_model, rf_model = None, None, None
try:
    with open(os.path.join(MODEL_PATH, "logistic_regression_model.pkl"), "rb") as f:
        lr_model = pickle.load(f)
    print("Logistic Regression model loaded successfully.")
except Exception as e:
    print(f"Error loading Logistic Regression model: {e}")

try:
    with open(os.path.join(MODEL_PATH, "gradient_boosting_model.pkl"), "rb") as f:
        gb_model = pickle.load(f)
    print("Gradient Boosting model loaded successfully.")
except Exception as e:
    print(f"Error loading Gradient Boosting model: {e}")

try:
    with open(os.path.join(MODEL_PATH, "random_forest_model.pkl"), "rb") as f:
        rf_model = pickle.load(f)
    print("Random Forest model loaded successfully.")
except Exception as e:
    print(f"Error loading Random Forest model: {e}")


# --- Data Configuration ---

# This list defines the exact column order that the models were trained on.
INPUT_COLS = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'Diabetes',
    'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare',
    'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age',
    'Education', 'Income'
]

# This dictionary provides a complete set of default values for all expected inputs.
ALL_DEFAULTS = {
    'HighBP': 0.0, 'HighChol': 0.0, 'CholCheck': 1.0, 'BMI': 25.0,
    'Smoker': 0.0, 'Stroke': 0.0, 'Diabetes': 0.0, 'PhysActivity': 1.0,
    'Fruits': 1.0, 'Veggies': 1.0, 'HvyAlcoholConsump': 0.0,
    'AnyHealthcare': 1.0, 'NoDocbcCost': 0.0, 'GenHlth': 3.0,
    'MentHlth': 0.0, 'PhysHlth': 0.0, 'DiffWalk': 0.0, 'Sex': 1.0,
    'Age': 8.0, 'Education': 4.0, 'Income': 6.0
}

# --- Flask Routes ---

@app.route("/")
def index():
    """Renders the main page with the prediction form."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handles the form submission, processes the data, and returns the prediction."""
    try:
        # --- FINAL ROBUST DATA PREPARATION ---
        user_input = request.form.to_dict()
        input_data = {}

        for col in INPUT_COLS:
            value = user_input.get(col)
            if value and value.strip():
                try:
                    input_data[col] = float(value)
                except (ValueError, TypeError):
                    print(f"Warning: Invalid value '{value}' for '{col}'. Using default.")
                    input_data[col] = ALL_DEFAULTS[col]
            else:
                input_data[col] = ALL_DEFAULTS[col]

        input_df = pd.DataFrame([input_data])[INPUT_COLS]
        
        print("--- Input Data for Model ---")
        print(input_df.to_string())
        print("----------------------------")

        # --- Model Selection and Prediction ---
        model_choice = user_input.get("model", "Random Forest") # Default to RF
        model_to_use = rf_model

        if model_choice == "Logistic Regression" and lr_model:
            model_to_use = lr_model
        elif model_choice == "Gradient Boosting" and gb_model:
            model_to_use = gb_model
        elif model_choice == "Random Forest" and rf_model:
            model_to_use = rf_model
        
        if not model_to_use:
            return render_template("result.html",
                                   prediction_text="Prediction Error",
                                   interpretation_message=f"The selected model ({model_choice}) is not available. It might have failed to load on startup.",
                                   confidence_score=None)

        prediction = model_to_use.predict(input_df)[0]
        probabilities = model_to_use.predict_proba(input_df)[0]

        # --- Prepare Results for Display ---
        if prediction == 1:
            prediction_text = "Heart Disease Detected"
            confidence_score = probabilities[1]
            interpretation_message = (
                "Based on the provided data, there is an indication of potential heart disease. "
                "It is crucial to consult a healthcare professional for a comprehensive diagnosis and personalized advice. "
                
            )
        else:
            prediction_text = "No Heart Disease Indicated"
            confidence_score = probabilities[0]
            interpretation_message = (
                "The prediction suggests a lower risk of heart disease based on your inputs. "
                "However, regular check-ups and maintaining a healthy lifestyle are always recommended. "
                
            )

        return render_template("result.html",
                               prediction_text=prediction_text,
                               confidence_score=f"{confidence_score:.2%}",
                               interpretation_message=interpretation_message)

    except Exception as e:
        print("--- AN UNEXPECTED ERROR OCCURRED ---")
        print(traceback.format_exc())
        print("------------------------------------")
        return render_template("result.html",
                               prediction_text="Prediction Error",
                               interpretation_message="An unexpected server error occurred. Please try again later.",
                               confidence_score=None)

@app.route("/dashboard")
def dashboard():
    """Renders the dashboard page."""
    return render_template("dashboard.html")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
