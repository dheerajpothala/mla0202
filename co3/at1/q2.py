import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

digits = load_digits()
X2, y2 = digits.data, digits.target
X2_scaled = StandardScaler().fit_transform(X2)

kmeans2 = KMeans(n_clusters=10, random_state=42, n_init=10)
kmeans_labels = kmeans2.fit_predict(X2_scaled)

gmm = GaussianMixture(n_components=10, random_state=42)
gmm_labels = gmm.fit_predict(X2_scaled)

kmeans_sil = silhouette_score(X2_scaled, kmeans_labels)
gmm_sil = silhouette_score(X2_scaled, gmm_labels)
kmeans_ari = adjusted_rand_score(y2, kmeans_labels)
gmm_ari = adjusted_rand_score(y2, gmm_labels)

pca2 = PCA(n_components=2)
X2_pca = pca2.fit_transform(X2_scaled)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(X2_pca[:, 0], X2_pca[:, 1], c=kmeans_labels, cmap="tab10", s=10)
axes[0].set_title("K-Means Clusters (PCA)")
axes[1].scatter(X2_pca[:, 0], X2_pca[:, 1], c=gmm_labels, cmap="tab10", s=10)
axes[1].set_title("GMM Clusters (PCA)")
plt.savefig("q2_clusters.png", dpi=100, bbox_inches="tight")
plt.close()

print("Dheeraj Krushna / 192525230")
print("K-Means Silhouette Score:", round(kmeans_sil, 4))
print("GMM Silhouette Score:", round(gmm_sil, 4))
print("K-Means Adjusted Rand Index:", round(kmeans_ari, 4))
print("GMM Adjusted Rand Index:", round(gmm_ari, 4))
