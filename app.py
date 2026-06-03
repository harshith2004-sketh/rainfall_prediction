# app.py
from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

model, feature_list = joblib.load("model/rainfall_model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        data = [float(request.form[feature]) for feature in feature_list]
        features = np.array(data).reshape(1, -1)
        prediction = model.predict(features)
        result = "Yes" if prediction[0] == 1 else "No"
        return render_template("index.html", prediction_text=f"Rain Tomorrow: {result}")
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
