import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis, FastICA

wine = load_wine()
X3, y3 = wine.data, wine.target
X3_scaled = StandardScaler().fit_transform(X3)

pca3 = PCA(n_components=2)
X3_pca = pca3.fit_transform(X3_scaled)

fa = FactorAnalysis(n_components=2, random_state=42)
X3_fa = fa.fit_transform(X3_scaled)

ica = FastICA(n_components=2, random_state=42, max_iter=1000)
X3_ica = ica.fit_transform(X3_scaled)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, comp, title in zip(axes, [X3_pca, X3_fa, X3_ica], ["PCA", "Factor Analysis", "ICA"]):
    ax.scatter(comp[:, 0], comp[:, 1], c=y3, cmap="viridis", s=15)
    ax.set_title(title)
plt.savefig("q3_components.png", dpi=100, bbox_inches="tight")
plt.close()

print("Dheeraj Krushna / 192525230")
print("Original feature count:", X3.shape[1])
print("PCA explained variance ratio:", pca3.explained_variance_ratio_)
print("PCA total variance retained:", round(sum(pca3.explained_variance_ratio_), 4))
print("FA components shape:", X3_fa.shape)
print("ICA components shape:", X3_ica.shape)
