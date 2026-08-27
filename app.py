from flask import Flask, request, jsonify, send_from_directory
import joblib
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "student_model.pkl"
)

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
scaler = model_data["scaler"]
kmeans = model_data["kmeans"]
label_encoder = model_data["label_encoder"]
features = model_data["features"]
results = model_data["results"]
model_name = model_data["model_name"]

app = Flask(__name__)


@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        attendance = float(data["attendance"])
        quiz_score = float(data["quizScore"])
        assignment_completion = float(data["assignmentCompletion"])
        engagement_score = float(data["engagementScore"])

        time_on_task = round(
            (
                attendance +
                quiz_score +
                assignment_completion +
                engagement_score
            ) / 400 * 10,
            2
        )

        student_data = pd.DataFrame(
            [[
                attendance,
                quiz_score,
                assignment_completion,
                time_on_task,
                engagement_score
            ]],
            columns=features
        )

        student_scaled = scaler.transform(student_data)

        prediction = model.predict(student_scaled)
        performance = label_encoder.inverse_transform(prediction)[0]

        probabilities = model.predict_proba(student_scaled)[0]
        confidence = float(max(probabilities) * 100)

        cluster_number = int(
            kmeans.predict(student_scaled)[0]
        )

        estimated_grade = (
            attendance * 0.30 +
            quiz_score * 0.35 +
            assignment_completion * 0.20 +
            engagement_score * 0.15
        )

        estimated_grade = round(
            min(max(estimated_grade, 0), 100),
            2
        )

        if performance == "High":
            risk_level = "LOW RISK"
        elif performance == "Medium":
            risk_level = "WATCH"
        else:
            risk_level = "AT RISK"

        return jsonify({
            "success": True,
            "performance": performance,
            "estimatedGrade": estimated_grade,
            "riskLevel": risk_level,
            "cluster": cluster_number + 1,
            "confidence": round(confidence, 2),
            "timeOnTask": time_on_task,
            "model": model_name,
            "modelResults": results
        })

    except Exception as error:
        return jsonify({
            "success": False,
            "error": str(error)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)