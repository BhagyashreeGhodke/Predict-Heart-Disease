import os
import pickle
import gdown
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Define model download URLs
MODEL_URLS = {
    "logistic_regression_model.pkl": "https://drive.google.com/uc?id=1yZLf84PhhNhCa1Y6TfRf0io8RN9t__LE",
    "random_forest_model.pkl": "https://drive.google.com/uc?id=12n2UT3zTFysga-ZVM5n3-sRJAN2r-lvl",
    "gradient_boosting_model.pkl": "https://drive.google.com/uc?id=1zGm5yaT_gykP2aqBw6KybB1l78MtvUsb",
}

# Set up model storage path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_PATH, exist_ok=True)

# Download models if missing
for model_name, url in MODEL_URLS.items():
    file_path = os.path.join(MODEL_PATH, model_name)
    if not os.path.exists(file_path):
        print(f"Downloading {model_name}...")
        try:
            gdown.download(url, file_path, quiet=False)
            print(f"Saved {model_name} to {file_path}")
        except Exception as e:
            print(f"Error downloading {model_name}: {e}")
            # You might want to exit or raise an error here if a model is critical
    else:
        print(f"{model_name} already exists.")

# Load models
try:
    with open(os.path.join(MODEL_PATH, "logistic_regression_model.pkl"), "rb") as f:
        lr_model = pickle.load(f)
    print("Logistic Regression model loaded.")

    with open(os.path.join(MODEL_PATH, "random_forest_model.pkl"), "rb") as f:
        rf_model = pickle.load(f)
    print("Random Forest model loaded.")

    with open(os.path.join(MODEL_PATH, "gradient_boosting_model.pkl"), "rb") as f:
        gb_model = pickle.load(f)
    print("Gradient Boosting model loaded.")

except FileNotFoundError as e:
    print(f"Error loading model file: {e}. Please ensure models are downloaded correctly.")
    lr_model, rf_model, gb_model = None, None, None # Set models to None if loading fails
    # In a production app, you might want to gracefully handle this, e.g., by showing a maintenance page

# Input columns (excluding the target: HeartDiseaseorAttack)
INPUT_COLS = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'Diabetes',
    'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare',
    'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age',
    'Education', 'Income'
]

# Default values for any missing or incorrect user input
default_values = {
    'Diabetes': 0, 'HighBP': 0, 'HighChol': 0, 'CholCheck': 1, 'BMI': 25.0,
    'Smoker': 0, 'Stroke': 0, 'PhysActivity': 1, 'Fruits': 1, 'Veggies': 1,
    'HvyAlcoholConsump': 0, 'AnyHealthcare': 1, 'NoDocbcCost': 0, 'GenHlth': 3,
    'MentHlth': 0, 'PhysHlth': 0, 'DiffWalk': 0, 'Sex': 1, 'Age': 8,
    'Education': 4, 'Income': 6
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        user_input = request.form.to_dict()
        input_data = {}

        for col in INPUT_COLS:
            raw_value = user_input.get(col, "").strip()
            try:
                # Convert to float; if empty or invalid, use default
                value = float(raw_value) if raw_value else default_values[col]
            except (ValueError, TypeError):
                value = default_values[col] # Fallback if conversion fails
            input_data[col] = value

        # Create DataFrame for prediction
        input_df = pd.DataFrame([input_data])

        model_choice = user_input.get("model", "Logistic Regression")
        model_to_use = None
        # model_name variable is no longer needed for the interpretation message

        if model_choice == "Logistic Regression":
            model_to_use = lr_model
        elif model_choice == "Random Forest":
            model_to_use = rf_model
        elif model_choice == "Gradient Boosting":
            model_to_use = gb_model
        else:
            # Handle invalid model choice
            return render_template("result.html",
                                   prediction_text="Error: Invalid model selected.",
                                   interpretation_message="Please go back and select a valid prediction model.",
                                   confidence_score=None)

        if model_to_use is None:
            # Handle case where model failed to load at startup
            return render_template("result.html",
                                   prediction_text="Error: Prediction model not available.",
                                   interpretation_message="The selected model could not be loaded. Please try again later or contact support.",
                                   confidence_score=None)

        # Perform prediction
        prediction = model_to_use.predict(input_df)[0]
        # Get probabilities for both classes [prob_class_0, prob_class_1]
        probabilities = model_to_use.predict_proba(input_df)[0]

        prediction_text = ""
        interpretation_message = ""
        confidence_score = None

        if prediction == 1:
            prediction_text = "Heart Disease Detected"
            # Confidence in having heart disease (class 1)
            confidence_score = probabilities[1]
            interpretation_message = (
                "Based on the provided data, there is an indication of potential heart disease. "
                "It is crucial to consult a healthcare professional for a comprehensive diagnosis and personalized advice. "
                "This prediction is for informational purposes only and should not replace professional medical consultation."
            )
        else:
            prediction_text = "No Heart Disease Indicated"
            # Confidence in NOT having heart disease (class 0)
            confidence_score = probabilities[0]
            interpretation_message = (
                "The prediction suggests a lower risk of heart disease based on your inputs. "
                "However, regular check-ups, maintaining a healthy lifestyle, and addressing any concerns with a healthcare professional are always recommended. "
                "This prediction is for informational purposes only and should not replace professional medical consultation."
            )

        return render_template("result.html",
                               prediction_text=prediction_text,
                               confidence_score=confidence_score,
                               interpretation_message=interpretation_message)

    except Exception as e:
        # Catch any unexpected errors during the prediction process
        print(f"An error occurred during prediction: {e}")
        return render_template("result.html",
                               prediction_text="Prediction Error",
                               interpretation_message=f"An unexpected error occurred. Please try again.",
                               confidence_score=None)

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

if __name__ == "__main__":
    app.run(debug=True)

