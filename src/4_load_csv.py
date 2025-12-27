# running code for clustering 

import pandas as pd  
import numpy as np  
from sklearn.preprocessing import StandardScaler  
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster  
from scipy.spatial.distance import pdist  
import matplotlib.pyplot as plt  

# Load and normalize features  
df = pd.read_csv('data/dataset_72_qawwali_features.csv', index_col='song_id')  
scaler = StandardScaler()  
df_scaled = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)  

# Compute distances and perform hierarchical clustering  
distances = pdist(df_scaled, metric='euclidean')  
Z = linkage(distances, method='ward')  

# Plot dendrogram  
plt.figure(figsize=(12, 6))  
dendrogram(Z, labels=df_scaled.index.tolist(), leaf_rotation=90)  
plt.title('Hierarchical Clustering Dendrogram')  
plt.tight_layout()  
plt.savefig('figures/dendrogram.png', dpi=300)  
plt.close()  

# Cut dendrogram at 4 clusters  
cluster_labels = fcluster(Z, t=4, criterion='maxclust')  
df_scaled['cluster'] = cluster_labels  

# Save clustered data  
df_clustered = df_scaled.copy()  
df_clustered.to_csv('data/dataset_72_features_clustered.csv')  
  
print(f'Cluster distribution:')  
print(df_clustered['cluster'].value_counts().sort_index())