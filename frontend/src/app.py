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
        gdown.download(url, file_path, quiet=False)
        print(f"Saved {model_name} to {file_path}")

# Load models
with open(os.path.join(MODEL_PATH, "logistic_regression_model.pkl"), "rb") as f:
    lr_model = pickle.load(f)

with open(os.path.join(MODEL_PATH, "random_forest_model.pkl"), "rb") as f:
    rf_model = pickle.load(f)

with open(os.path.join(MODEL_PATH, "gradient_boosting_model.pkl"), "rb") as f:
    gb_model = pickle.load(f)

# Input columns (excluding the target: HeartDiseaseorAttack)
# CORRECT: without target
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
                value = float(raw_value)
            except (ValueError, TypeError):
                value = default_values[col]
            input_data[col] = value

        # Create DataFrame for prediction
        input_df = pd.DataFrame([input_data])

        model_choice = user_input.get("model", "Logistic Regression")

        if model_choice == "Logistic Regression":
            prediction = lr_model.predict(input_df)[0]
            model_used = "Logistic Regression"
        elif model_choice == "Random Forest":
            prediction = rf_model.predict(input_df)[0]
            model_used = "Random Forest"
        elif model_choice == "Gradient Boosting":
            prediction = gb_model.predict(input_df)[0]
            model_used = "Gradient Boosting"
        else:
            return render_template("result.html", prediction_text="Invalid model selected.")

        result = "Heart Disease" if prediction == 1 else "No Heart Disease"
        result_text = f"Prediction: {result} using {model_used}"

        return render_template("result.html", prediction_text=result_text)

    except Exception as e:
        return f"Error: {str(e)}"


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


if __name__ == "__main__":
    app.run(debug=True)
