import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

np.random.seed(42)
n = 200
data = pd.DataFrame({
    "CustomerID": range(1, n + 1),
    "Gender": np.random.choice(["Male", "Female"], n),
    "Age": np.random.randint(18, 70, n),
    "Annual Income (k$)": np.random.randint(15, 140, n),
    "Spending Score (1-100)": np.random.randint(1, 100, n)
})

X = data[["Annual Income (k$)", "Spending Score (1-100)"]].values
X_scaled = StandardScaler().fit_transform(X)

wcss = []
sil_scores = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    wcss.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

best_k = K_range[np.argmax(sil_scores)]
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_scaled)
data["Cluster"] = clusters

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(6, 4))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis")
plt.title("Customer Segments (PCA-reduced)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("q1_clusters.png", dpi=100, bbox_inches="tight")
plt.close()

print("Dheeraj Krushna / 192525230")
print("Elbow WCSS values:", [round(v, 2) for v in wcss])
print("Silhouette scores:", [round(v, 3) for v in sil_scores])
print("Best k by silhouette score:", best_k)
print("Cluster counts:\n", data["Cluster"].value_counts().sort_index())
print("Explained variance ratio (PCA):", pca.explained_variance_ratio_)

# OUTPUT:
# Dheeraj Krushna / 192525230
# Elbow WCSS values: [241.1, 148.33, 92.98, 74.48, 60.73, 49.93, 41.63, 35.47, 30.94]
# Silhouette scores: [0.376, 0.412, 0.448, 0.437, 0.41, 0.407, 0.413, 0.415, 0.432]
# Best k by silhouette score: 4
# Cluster counts:
#  Cluster
#  0    64
#  1    50
#  2    27
#  3    59
#  Name: count, dtype: int64
# Explained variance ratio (PCA): [0.53274478 0.46725522]
