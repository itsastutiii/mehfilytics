# Load the Clustered, Scaled Data
import pandas as pd

# Load clustered dataset (already z-scored)
df = pd.read_csv(
    "data/qawwali_features_clustered.csv",
    index_col="song_id"
)

X = df.drop(columns=["cluster"])
y = df["cluster"]

print(X.shape, y.shape)

# o/p: (34, 8) (34,)

# Run PCA (2 Components Only)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total explained variance:", pca.explained_variance_ratio_.sum())

# Create & Save PCA Scatter Plot
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

plt.figure(figsize=(7, 6))

for cluster_id in sorted(y.unique()):
    mask = y == cluster_id
    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=f"Cluster {cluster_id}",
        s=60,
        alpha=0.8
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection of Qawwali Performances")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("figures/pca_clusters.png", dpi=300)
plt.close()

print("Saved: figures/pca_clusters.png")

# Label Points
plt.figure(figsize=(8, 7))

for i, song_id in enumerate(X.index):
    plt.scatter(X_pca[i, 0], X_pca[i, 1], c=f"C{y.iloc[i]-1}")
    plt.text(
        X_pca[i, 0] + 0.03,
        X_pca[i, 1] + 0.03,
        song_id,
        fontsize=8
    )

plt.title("PCA with Song IDs")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()

plt.savefig("figures/pca_clusters_labeled.png", dpi=300)
plt.close()

print("Saved: figures/pca_clusters_labeled.png")

# OUTPUT: 
# Explained variance ratio: [0.33342327 0.26904717]
#Exp: 
# ~33% + ~27% = ~60% total variance
# For audio features + unsupervised data, this is very strong
# PCA is not collapsing everything into noise
# But also not perfectly separating clusters (which would be suspicious)

# Total explained variance: 0.6024704353504137
# Saved: figures/pca_clusters.png
# Saved: figures/pca_clusters_labeled.png

