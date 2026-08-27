import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "student_data.csv")

df = pd.read_csv(DATA_PATH)

features = [
    "Attendance",
    "QuizScore",
    "AssignmentCompletion",
    "TimeOnTask",
    "EngagementScore"
]

latest = df.sort_values("Week").groupby("StudentID").tail(1).copy()

X = latest[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

latest["PC1"] = X_pca[:, 0]
latest["PC2"] = X_pca[:, 1]
latest["Cluster"] = clusters + 1

cluster_means = latest.groupby("Cluster")[features].mean().mean(axis=1)

cluster_order = cluster_means.sort_values().index.tolist()

cluster_names = {
    cluster_order[0]: "Low-profile learners",
    cluster_order[1]: "Moderate-profile learners",
    cluster_order[2]: "High-profile learners"
}

plt.figure(figsize=(10, 7))

for cluster in sorted(latest["Cluster"].unique()):
    subset = latest[latest["Cluster"] == cluster]

    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        label=f"Cluster {cluster}: {cluster_names[cluster]}"
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Student Learning Pattern Clustering using PCA")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

output_path = os.path.join(
    BASE_DIR,
    "model",
    "pca_clustering.png"
)

plt.savefig(output_path, dpi=300)
plt.show()

print("Students plotted:", len(latest))
print("PCA explained variance:", pca.explained_variance_ratio_)
print("Total explained variance:", pca.explained_variance_ratio_.sum())
print("Cluster distribution:")
print(latest["Cluster"].value_counts().sort_index())
print("Visualization saved at:", output_path)