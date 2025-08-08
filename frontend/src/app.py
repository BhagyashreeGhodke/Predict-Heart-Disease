import os
import pickle
import gdown
import pandas as pd
from flask import Flask, render_template, request
import numpy as np
import traceback
import db
from dotenv import load_dotenv
from pathlib import Path 


# --- Initialization---
# This code navigates up three levels from app.py (src -> frontend -> root) to find the .env file.
dotenv_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)
print(f"DATABASE_URL: {os.getenv('DATABASE_URL')}")

# Initialize the Flask application
app = Flask(__name__)
DATABASE_URL = os.getenv('DATABASE_URL')
print("DATABASE_URL: ",DATABASE_URL)
# --- Database Initialization (Conditional) ---
# Only initialize the database if the DATABASE_URL is set
if DATABASE_URL:
    print("DATABASE_URL found. Initializing database...")
    try:
        db.initialize_database()
    except Exception as db_init_error:
        print(f"--- WARNING: Database initialization failed ---")
        print(f"Error: {db_init_error}")
        print("---------------------------------------------")
else:
    print("DATABASE_URL not found. Skipping database initialization.")


# --- Model Downloading and Loading ---

# Define the Google Drive URL for the Random Forest model
#MODEL_URL = "https://drive.google.com/uc?id=1VIjQV2f04CRQrQv-xgef7M81SinBiXTD"
MODEL_URL = "https://drive.google.com/uc?id=1ykAtYjp0DmOg7MB0-rdPBdyK6lz7Iy_c"
MODEL_NAME = "Random_forest.pkl"

# Set up a local path to store the model

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_PATH, exist_ok=True)

# Download the model from Google Drive if it doesn't exist locally
file_path = os.path.join(MODEL_PATH, MODEL_NAME)
if not os.path.exists(file_path):
    print(f"Downloading {MODEL_NAME}...")
    try:
        gdown.download(MODEL_URL, file_path, quiet=False)
        print(f"Successfully saved {MODEL_NAME} to {file_path}")
    except Exception as e:
        print(f"Error downloading {MODEL_NAME}: {e}")
else:
    print(f"{MODEL_NAME} already exists locally.")

# Load the downloaded model into memory
model = None
try:
    with open(file_path, "rb") as f:
        model = pickle.load(f)
    print(f"{MODEL_NAME} loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")


# --- Data Configuration ---

# This list defines the exact column order that the model was trained on.
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
    """Handles the form submission, processes data, and returns the prediction."""
    try:
        # --- Robust Data Preparation ---
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

        # --- Model Prediction ---
        if not model:
            return render_template("result.html",
                                   prediction_text="Prediction Error",
                                   interpretation_message="The prediction model is not available. It might have failed to load on startup.",
                                   confidence_score=None)

        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        
        # --- Prepare Results for Display ---
        if prediction == 1:
            prediction_text = "Cardiovascular disease Detected"
            confidence_score = probabilities[1]
            interpretation_message = (
                "Based on the provided data, there is an indication of potential cardiovascular disease. "
                "It is crucial to consult a healthcare professional for a comprehensive diagnosis and personalized advice."
            )
        else:
            prediction_text = "No Cardiovascular disease Indicated"
            confidence_score = probabilities[0]
            interpretation_message = (
                "The prediction suggests a lower risk of cardiovascular disease based on your inputs. "
                "However, regular check-ups and maintaining a healthy lifestyle are always recommended."
            )
        
        confidence_score_formatted = f"{confidence_score:.2%}"
        
        # --- Save to Database (if possible) ---
        # This block will only run if the DATABASE_URL is available in the environment.
        try:
            if os.environ.get('DATABASE_URL'):
                db.save_prediction_to_db(input_data, prediction_text, confidence_score_formatted)
                print("Prediction saved to database.")
            else:
                print("DATABASE_URL not found. Skipping database save.")
        except Exception as db_error:
            # If saving fails for any reason, just print a warning but don't crash the app.
            print(f"--- WARNING: Could not save to database ---")
            print(f"Error: {db_error}")
            print("-------------------------------------------")


        return render_template("result.html",
                               prediction_text=prediction_text,
                               confidence_score=confidence_score_formatted,
                               interpretation_message=interpretation_message)

    except Exception as e:
        # --- ENHANCED FOR DEBUGGING ---
        # This will now display the actual error on the results page.
        print("--- AN UNEXPECTED ERROR OCCURRED ---")
        error_traceback = traceback.format_exc()
        print(error_traceback)
        print("------------------------------------")
        # Perform the replacement before the f-string to avoid syntax errors
        formatted_traceback = error_traceback.replace('\n', '<br>')
        return render_template("result.html",
                               prediction_text="Prediction Error",
                               interpretation_message=f"An unexpected error occurred: {str(e)}<br><br>Traceback:<br>{formatted_traceback}",
                               confidence_score=None)

@app.route("/dashboard")
def dashboard():
    """Renders the dashboard page."""
    return render_template("dashboard.html")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
