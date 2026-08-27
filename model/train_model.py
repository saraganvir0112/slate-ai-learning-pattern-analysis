import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "student_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "student_model.pkl")

df = pd.read_csv(DATA_PATH)

features = [
    "Attendance",
    "QuizScore",
    "AssignmentCompletion",
    "TimeOnTask",
    "EngagementScore"
]

X = df[features]
y = df["Performance"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)
kmeans.fit(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=6
    ),
    "Naive Bayes": GaussianNB()
}

results = {}
best_model = None
best_model_name = None
best_f1 = -1

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),
        "f1_score": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )
    }

    results[name] = metrics

    if metrics["f1_score"] > best_f1:
        best_f1 = metrics["f1_score"]
        best_model = model
        best_model_name = name

model_data = {
    "model": best_model,
    "model_name": best_model_name,
    "scaler": scaler,
    "pca": pca,
    "kmeans": kmeans,
    "label_encoder": label_encoder,
    "features": features,
    "results": results
}

joblib.dump(model_data, MODEL_PATH)

print("Dataset rows:", len(df))
print("Students:", df["StudentID"].nunique())
print("Weeks:", df["Week"].nunique())
print("Best model:", best_model_name)
print("Best F1:", round(best_f1, 4))
print(
    "PCA variance:",
    np.round(pca.explained_variance_ratio_, 4)
)
