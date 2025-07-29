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

# Define Google Drive URLs for the pre-trained models
MODEL_URLS = {
    "logistic_regression_model.pkl": "https://drive.google.com/uc?id=1yZLf84PhhNhCa1Y6TfRf0io8RN9t__LE",
    "gradient_boosting_model.pkl": "https://drive.google.com/uc?id=1zGm5yaT_gykP2aqBw6KybB1l78MtvUsb",
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
try:
    with open(os.path.join(MODEL_PATH, "logistic_regression_model.pkl"), "rb") as f:
        lr_model = pickle.load(f)
    print("Logistic Regression model loaded successfully.")

    with open(os.path.join(MODEL_PATH, "gradient_boosting_model.pkl"), "rb") as f:
        gb_model = pickle.load(f)
    print("Gradient Boosting model loaded successfully.")
    
except FileNotFoundError as e:
    print(f"FATAL: Could not load a model file: {e}. The application cannot make predictions.")
    lr_model, gb_model = None, None

# --- Data Configuration ---

# This list defines the exact column order that the models were trained on.
# This order is critical for the model to work correctly.
INPUT_COLS = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'Diabetes',
    'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare',
    'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age',
    'Education', 'Income'
]

# This dictionary provides a complete set of default values for all expected inputs.
# This makes the data preparation robust against missing or empty form fields.
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
        # --- EVEN MORE ROBUST DATA PREPARATION ---

        # 1. Get the data submitted by the user.
        user_input = request.form.to_dict()

        # 2. Create a DataFrame with one row of default values, ensuring all columns are present in the correct order.
        input_df = pd.DataFrame([ALL_DEFAULTS], columns=INPUT_COLS)

        # 3. Iterate through the user's submitted data and update the DataFrame.
        for col, value in user_input.items():
            # Only process columns that are expected by the model.
            if col in INPUT_COLS:
                # Check if the submitted value is valid (not None or just whitespace).
                if value and value.strip():
                    try:
                        # Update the specific cell in the DataFrame.
                        input_df.loc[0, col] = float(value)
                    except (ValueError, TypeError):
                        # If conversion to float fails, the default value from the initial DataFrame remains.
                        print(f"Warning: Invalid value '{value}' for column '{col}'. Using default.")
                        pass # The default is already in place, so no action is needed.
        
        print("--- Input Data for Model ---")
        print(input_df.to_string())
        print("----------------------------")

        # --- Model Selection and Prediction ---
        model_choice = user_input.get("model", "Logistic Regression")
        model_to_use = None

        if model_choice == "Logistic Regression" and lr_model:
            model_to_use = lr_model
        elif model_choice == "Gradient Boosting" and gb_model:
            model_to_use = gb_model
        
        if not model_to_use:
            return render_template("result.html",
                                   prediction_text="Prediction Error",
                                   interpretation_message="The selected model is not available. It might have failed to load on startup.",
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
                "This prediction is for informational purposes only."
            )
        else:
            prediction_text = "No Heart Disease Indicated"
            confidence_score = probabilities[0]
            interpretation_message = (
                "The prediction suggests a lower risk of heart disease based on your inputs. "
                "However, regular check-ups and maintaining a healthy lifestyle are always recommended. "
                "This prediction is for informational purposes only."
            )

        return render_template("result.html",
                               prediction_text=prediction_text,
                               confidence_score=confidence_score,
                               interpretation_message=interpretation_message)

    except Exception as e:
        # Catch any other unexpected errors and display a helpful message.
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
