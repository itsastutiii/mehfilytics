# phase 4 part 2
# Load the Clustered Dataset (Canonical)
import pandas as pd

# Load clustered feature table
# i dont like how all these are hardcoded. 
# a better way would probably be to replace all these with variables and have those variables stored in a central place like how we store API keys in .env, 
# and have them change, nothing else in the entire program needs to move around then, just those values. 

df = pd.read_csv(
    "data/dataset_72_features_clustered.csv",
    index_col="song_id"
)

print(df.head())
print(df["cluster"].value_counts())

# Compute Cluster Centroids
cluster_means = df.groupby("cluster").mean()
print(cluster_means)

# saveing centroids alag se 
cluster_means.to_csv("data/dataset_72_cluster_centroids.csv")

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

    out_path = f"figures/clusters/dataset_72_cluster_{cluster_id}_profile.png"
    plt.savefig(out_path, dpi=300)
    plt.close()   # IMPORTANT: prevents overlap / memory issues

    print(f"Saved: {out_path}")
