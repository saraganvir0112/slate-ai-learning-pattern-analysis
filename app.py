from flask import Flask, request, jsonify, send_from_directory
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "student_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "dataset", "student_data.csv")

model_data = joblib.load(MODEL_PATH)
weekly_data = pd.read_csv(DATA_PATH)

model = model_data["model"]
scaler = model_data["scaler"]
kmeans = model_data["kmeans"]
label_encoder = model_data["label_encoder"]
features = model_data["features"]
results = model_data["results"]
model_name = model_data["model_name"]

app = Flask(__name__)


def get_trajectory_pattern(student_df):
    ordered = student_df.sort_values("Week")
    scores = (
        ordered["Attendance"] * 0.30
        + ordered["QuizScore"] * 0.35
        + ordered["AssignmentCompletion"] * 0.20
        + ordered["EngagementScore"] * 0.15
    ).to_numpy()

    weeks = ordered["Week"].to_numpy().reshape(-1, 1)

    if len(scores) < 4:
        return "Plateaued"

    midpoint = len(scores) // 2

    early_x = weeks[:midpoint]
    late_x = weeks[midpoint:]

    early_slope = LinearRegression().fit(
        early_x,
        scores[:midpoint]
    ).coef_[0]

    late_slope = LinearRegression().fit(
        late_x,
        scores[midpoint:]
    ).coef_[0]

    overall_slope = LinearRegression().fit(
        weeks,
        scores
    ).coef_[0]

    first_half = scores[:midpoint].mean()
    second_half = scores[midpoint:].mean()

    if early_slope <= 0.8 and late_slope >= 2.0:
        return "Late bloomer"

    if overall_slope >= 1.6:
        return "Steady climber"

    if abs(second_half - first_half) < 4.5 and abs(overall_slope) < 0.8:
        return "Plateaued"

    if overall_slope <= -1.4:
        return "At-risk decliner"

    if late_slope > early_slope + 1.0:
        return "Late bloomer"

    if overall_slope > 0.8:
        return "Steady climber"

    if overall_slope < -0.8:
        return "At-risk decliner"

    return "Plateaued"


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
                attendance
                + quiz_score
                + assignment_completion
                + engagement_score
            ) / 400 * 10,
            2
        )

        input_data = pd.DataFrame(
            [[
                attendance,
                quiz_score,
                assignment_completion,
                time_on_task,
                engagement_score
            ]],
            columns=features
        )

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)
        performance = label_encoder.inverse_transform(prediction)[0]

        probabilities = model.predict_proba(input_scaled)[0]
        confidence = float(probabilities.max() * 100)

        cluster_number = int(kmeans.predict(input_scaled)[0])

        estimated_grade = (
            attendance * 0.30
            + quiz_score * 0.35
            + assignment_completion * 0.20
            + engagement_score * 0.15
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


@app.route("/students", methods=["GET"])
def students():
    try:
        output = []

        for student_id, group in weekly_data.groupby("StudentID"):
            ordered = group.sort_values("Week")
            latest = ordered.iloc[-1]

            current_data = pd.DataFrame(
                [[
                    latest["Attendance"],
                    latest["QuizScore"],
                    latest["AssignmentCompletion"],
                    latest["TimeOnTask"],
                    latest["EngagementScore"]
                ]],
                columns=features
            )

            current_scaled = scaler.transform(current_data)

            prediction = model.predict(current_scaled)
            performance = label_encoder.inverse_transform(prediction)[0]

            probabilities = model.predict_proba(current_scaled)[0]
            confidence = float(probabilities.max() * 100)

            cluster = int(kmeans.predict(current_scaled)[0]) + 1

            predicted = int(round(
                latest["Attendance"] * 0.30
                + latest["QuizScore"] * 0.35
                + latest["AssignmentCompletion"] * 0.20
                + latest["EngagementScore"] * 0.15
            ))

            if predicted >= 85:
                risk = "On track"
            elif predicted >= 70:
                risk = "Watch"
            elif predicted >= 55:
                risk = "At risk"
            else:
                risk = "High risk"

            pattern = get_trajectory_pattern(ordered)

            trend = (
                ordered["Attendance"] * 0.30
                + ordered["QuizScore"] * 0.35
                + ordered["AssignmentCompletion"] * 0.20
                + ordered["EngagementScore"] * 0.15
            ).round(1).tolist()

            output.append({
                "student": student_id,
                "attendance": int(round(latest["Attendance"])),
                "performance": performance,
                "predicted": predicted,
                "risk": risk,
                "confidence": round(confidence, 1),
                "cluster": cluster,
                "pattern": pattern,
                "trend": trend
            })

        risk_priority = {
            "High risk": 0,
            "At risk": 1,
            "Watch": 2,
            "On track": 3
        }

        output.sort(
            key=lambda student: (
                risk_priority[student["risk"]],
                student["predicted"]
            )
        )

        output = output[:6]

        return jsonify({
            "success": True,
            "students": output
        })

    except Exception as error:
        return jsonify({
            "success": False,
            "error": str(error)
        }), 400


@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "success": True,
        "model": model_name,
        "results": results,
        "students": int(weekly_data["StudentID"].nunique()),
        "weeks": int(weekly_data["Week"].nunique()),
        "records": int(len(weekly_data))
    })


if __name__ == "__main__":
    app.run(debug=True)
