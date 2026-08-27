import os
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "dataset", "student_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "student_model.pkl")

data = pd.read_csv(DATA_PATH)
model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
scaler = model_data["scaler"]
label_encoder = model_data["label_encoder"]
features = model_data["features"]
model_name = model_data["model_name"]

X = data[features]
y = label_encoder.transform(data["Performance"])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_test_scaled = scaler.transform(X_test)

predictions = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)
recall = recall_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)
f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

print("\nMODEL EVALUATION")
print("=" * 50)
print("Model:", model_name)
print("Dataset Size:", len(data))
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

print("\nPerformance Metrics")
print("-" * 50)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report")
print("-" * 50)

print(
    classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)

print("\nConfusion Matrix")
print("-" * 50)
print(confusion_matrix(y_test, predictions))