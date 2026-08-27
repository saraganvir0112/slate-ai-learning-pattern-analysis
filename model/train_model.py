import os
import pandas as pd
import joblib

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


print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("\nDataset Preview:")
print(df.head())

print("\nDataset Shape:", df.shape)


FEATURES = [
    "Attendance",
    "QuizScore",
    "AssignmentCompletion",
    "TimeOnTask",
    "EngagementScore",
]

TARGET = "Performance"

X = df[FEATURES]
y = df[TARGET]


label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nPerformance Classes:")
for index, label in enumerate(label_encoder.classes_):
    print(f"{index} = {label}")


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\nPCA Explained Variance:")
print(pca.explained_variance_ratio_)


kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

clusters = kmeans.fit_predict(X_scaled)

print("\nK-Means clustering completed.")


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)

models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "Naive Bayes": GaussianNB(),
}

results = {}

best_model = None
best_model_name = None
best_f1_score = 0

print("\n" + "=" * 55)
print("MODEL PERFORMANCE")
print("=" * 55)

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    recall = recall_score(y_test, predictions, average="weighted", zero_division=0)

    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

    results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    print(f"\n{name}")
    print("-" * 30)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    if f1 > best_f1_score:
        best_f1_score = f1
        best_model = model
        best_model_name = name

model_data = {
    "model": best_model,
    "model_name": best_model_name,
    "scaler": scaler,
    "pca": pca,
    "kmeans": kmeans,
    "label_encoder": label_encoder,
    "features": FEATURES,
    "results": results,
}

joblib.dump(model_data, MODEL_PATH)

print("\n" + "=" * 55)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 55)

print(f"Best Model: {best_model_name}")
print(f"Best F1 Score: {best_f1_score:.4f}")

print(f"\nModel saved at:")
print(MODEL_PATH)
