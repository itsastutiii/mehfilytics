# phase 4 
# Load the Clustered Dataset (Canonical)
import pandas as pd

# Load clustered feature table
df = pd.read_csv(
    "data/qawwali_features_clustered.csv",
    index_col="song_id"
)

print(df.head())
print(df["cluster"].value_counts())

# Compute Cluster Centroids
cluster_means = df.groupby("cluster").mean()
print(cluster_means)

# saveing centroids alag se 
cluster_means.to_csv("data/cluster_centroids.csv")

# Make the Differences Explicit (Readable Table)
cluster_means_rounded = cluster_means.round(2)
print(cluster_means_rounded)

cluster_ranks = cluster_means.rank(axis=1, ascending=False)
print(cluster_ranks)

# visualising cluster profiles 
import matplotlib.pyplot as plt

for cluster_id in cluster_means.index:
    cluster_means.loc[cluster_id].plot(
        kind="bar",
        figsize=(8, 4),
        title=f"Cluster {cluster_id} – Feature Profile"
    )
    plt.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.show()

# OUTPUT: 
# 🟣 Cluster 1 — Vocal / Taali-Forward
# High taali activity
# Low tabla energy
# 🔴 Cluster 2 — Percussion-Dominant
# Highest tabla energy & variance
# Moderate taali
# 🟢 Cluster 3 — Restrained / Low-Energy
# Low everything
# Transitional / meditative sections
# 🔵 Cluster 4 — Steady Rhythmic Grounding
# Near-average tabla
# Very low taali
# Minimal expressive surges

# save cluste figures 
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

for cluster_id in cluster_means.index:
    plt.figure(figsize=(8, 4))

    cluster_means.loc[cluster_id].plot(
        kind="bar"
    )

    plt.axhline(0, color="black", linewidth=0.8)
    plt.title(f"Cluster {cluster_id} – Feature Profile")
    plt.ylabel("Z-score (relative to dataset mean)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out_path = f"figures/cluster_{cluster_id}_profile.png"
    plt.savefig(out_path, dpi=300)
    plt.close()   # IMPORTANT: prevents overlap / memory issues

    print(f"Saved: {out_path}")
