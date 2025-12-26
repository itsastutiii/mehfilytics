# all of this would be a lot easier and organised if REPL ya kuch type CLI ho 
# if not, sequentially script execute kar do idhar 

import pandas as pd

df = pd.read_csv("data/qawwali_features.csv", index_col="song_id")

# print(df.head())
# print("\nShape:", df.shape)

# pd.set_option("display.float_format", "{:.4f}".format)
# print(df.describe())

#Z SCORE NORMALISATION 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df_scaled = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns,
    index=df.index
)

# print(df_scaled.head())
# print("\nMeans:")
# print(df_scaled.mean())
# print("\nStd Devs:")
# print(df_scaled.std())

# corr = df_scaled.corr()
# print(corr)

#VISUALISATION EXTRA
# import seaborn as sns
# import matplotlib.pyplot as plt

# sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Feature Correlation Matrix")
# plt.show()


from scipy.spatial.distance import pdist, squareform

# df_scaled already exists from Phase 3.5
distance_matrix = pdist(df_scaled.values, metric="euclidean")

from scipy.cluster.hierarchy import linkage

Z = linkage(distance_matrix, method="ward")

from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
dendrogram(
    Z,
    labels=df_scaled.index.tolist(),
    leaf_rotation=90
)
plt.title("Hierarchical Clustering of Qawwali Performances")
plt.xlabel("Song ID")
plt.ylabel("Euclidean Distance")
plt.tight_layout()
# plt.show()

#now we cut the dendogram 
from scipy.cluster.hierarchy import fcluster

# Cut to get 4 clusters
cluster_labels = fcluster(Z, t=4, criterion="maxclust")

df_clustered = df_scaled.copy()
df_clustered["cluster"] = cluster_labels

# print(df_clustered["cluster"].value_counts())


# saving the same 
# p.s. A cluster represents a group of performances with similar acoustic performance characteristics, not a genre, raag, or semantic category.

# df_clustered = df_scaled.copy()
# df_clustered["cluster"] = cluster_labels
# df_clustered.to_csv("data/qawwali_features_clustered.csv")


# print(df_clustered.groupby("cluster").mean())
#output: 
#          tabla_energy_mean  tabla_energy_var  ...  taali_activity  taali_burstiness
# cluster                                       ...                                  
# 1                -0.203731         -0.330521  ...        1.652609          0.100392
# 2                 1.467777          1.631629  ...        0.348527         -0.035471
# 3                -0.654723         -0.493684  ...       -0.384895          0.099720
# 4                 0.027239         -0.188243  ...       -0.738808         -0.158617

# [4 rows x 8 columns]

