# src/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data
df = pd.read_csv("data/weatherAUS.csv")
df.dropna(subset=["RainTomorrow"], inplace=True)
df.fillna(method='ffill', inplace=True)

# Encode categorical columns
df = df.drop(columns=["Date", "Location"])
df = df.astype(str).apply(lambda col: pd.factorize(col)[0] if col.dtype == 'object' else col)

# Select features (you can adjust this list)
features = ["MinTemp", "MaxTemp", "Rainfall", "Humidity9am", "Humidity3pm",
            "Pressure9am", "Pressure3pm", "Temp3pm", "RainToday", "WindGustDir"]

X = df[features]
y = pd.factorize(df["RainTomorrow"])[0]  # Encode target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and feature list
joblib.dump((model, features), "model/rainfall_model.pkl")
