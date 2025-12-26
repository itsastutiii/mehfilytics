# Complete Step-by-Step Technical Implementation Guide
## Phase 0 — Environment & Repository Setup
Purpose: Set up Python environment and install required dependencies for audio processing and Big Data analytics.

### 1. Create Python virtual environment
```
python3 -m venv qawwalrang_env  
source qawwalrang_env/bin/activate
```
### 2. Install dependencies
```
pip install numpy pandas librosa matplotlib scipy scikit-learn pyspark yt-dlp gdown ffmpy
```
### 3. Verify repository structure
```
ls -la src/  
ls -la data/  
ls -la metadata/
```
### 4. Cross check imports
```
python -c "import librosa, numpy, pandas, sklearn, pyspark; print('All imports successful')"
```
**Files used** : `src/` directory scripts,` metadata/` JSON files\
**Outputs generated**: Activated virtual environment\
**Verification**: All Python imports succeed without errors

## Phase 1 — Dataset Reconstruction & Curation
Purpose: Download and curate raw Qawwali audio files from metadata specifications.

### 1. Navigate to project root and run dataset builder
```
python src/qdsb.py data/qawwali_audio_clean metadata/metadata_of_retrieved.json
```
### 2. Verify audio downloads
```
ls -la data/qawwali_audio_clean/*.mp3 | wc -l  
# Should show 34 files (atleast at time of exe for me i could extract 34 :P)
```
### 3. Check audio format consistency
```
python -c "  
import librosa  
import glob  
for f in glob.glob('data/qawwali_audio_clean/*.mp3')[:5]:  
    y, sr = librosa.load(f, sr=None)  
    print(f'{f}: {sr}Hz, mono={len(y.shape)==1}, duration={len(y)/sr:.2f}s')  
"
```
### 4. Validate songs-data.npy creation
```
python -c "  
import numpy as np  
data = np.load('data/qawwali_audio_clean/songs-data.npy', allow_pickle=True).item()  
print(f'Songs loaded: {len([k for k in data.keys() if k != \"rate\"])}')  
print(f'Sample rate: {data[\"rate\"]}Hz')  
"
```
Files used: `src/qdsb.py`, `metadata/metadata_of_retrieved.json`\
Outputs generated:
```
data/qawwali_audio_clean/*.mp3 (34 files)
data/qawwali_audio_clean/songs-data.npy
```
Verification: 34 audio files at 44.1kHz mono, ~60 seconds each

## Phase 2 — Signal-Level Feature Extraction
Purpose: Extract tabla (CQT) and taali (MFCC) features from audio using source separation.

### 1. Run feature extraction with reload and extract flags
```
python src/qdetect.py data/qawwali_audio_clean --reload --extract
```

### 2. Verify feature file creation
```
ls -la data/qawwali_audio_clean/features/
```

### 3. Check feature matrix shapes
```
python -c "  
import numpy as np  
features = np.load('data/qawwali_audio_clean/features/tt-features.npy', allow_pickle=True).item()  
for key, mat in list(features.items())[:4]:  
    print(f'{key}: shape={mat.shape}, type={type(mat)}')  
"
```
### 4. Validate time-frequency representations
```
python -c "  
import numpy as np  
features = np.load('data/qawwali_audio_clean/features/tt-features.npy', allow_pickle=True).item()  
tabla_keys = [k for k in features.keys() if 'tabla' in k]  
taali_keys = [k for k in features.keys() if 'taali' in k]  
print(f'Tabla features: {len(tabla_keys)} files')  
print(f'Taali features: {len(taali_keys)} files')  
print(f'Tabla CQT bins: {features[tabla_keys[0]].shape[0]} (expected: 84)')  
print(f'Taali MFCC coeffs: {features[taali_keys[0]].shape[0]} (expected: 13)')  
"
```
Files used: `src/qdetect.py`\
Outputs generated:
```
data/qawwali_audio_clean/features/tt-features.npy
```
Verification: Tabla features have 84 CQT bins, Taali features have 13 MFCC coefficients, both are time-frequency matrices

## Phase 3 — Feature Reduction & Validation
Purpose: Reduce time-frequency matrices to interpretable per-song scalar features.

### 1. Run feature reduction script
```
python src/4_reduce_features.py
```
### 2. Verify CSV output structure
```
head -5 data/qawwali_features.csv  
wc -l data/qawwali_features.csv
```
### 3. Check feature dimensions and ranges
```
python -c "  
import pandas as pd  
df = pd.read_csv('data/qawwali_features.csv', index_col='song_id')  
print(f'Data shape: {df.shape}')  
print(f'Feature columns: {list(df.columns)}')  
print(f'Feature ranges:')  
print(df.describe().loc[['min', 'max', 'mean']])  
"
```
### 4. Validate feature correlations
```
python -c "  
import pandas as pd  
import numpy as np  
df = pd.read_csv('data/qawwali_features.csv', index_col='song_id')  
corr = df.corr()  
print('Feature correlations (absolute values > 0.5):')  
for i in range(len(corr.columns)):  
    for j in range(i+1, len(corr.columns)):  
        if abs(corr.iloc[i, j]) > 0.5:  
            print(f'{corr.columns[i]} - {corr.columns[j]}: {corr.iloc[i, j]:.3f}')  
"
```
Files used: `src/4_reduce_features.py`\ 
Outputs generated: `data/qawwali_features.csv`\
Verification: 34 rows × 8 features, reasonable ranges, partial independence between tabla and taali features

## Phase 4A — Unsupervised Discovery (Hierarchical Clustering)
Purpose: Apply hierarchical clustering to discover performance regimes in feature space.

### 1. Create clustering script (since 5_load_csv.py is referenced but not provided)
```
cat > src/5_load_csv.py << 'EOF'  
import pandas as pd  
import numpy as np  
from sklearn.preprocessing import StandardScaler  
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster  
from scipy.spatial.distance import pdist  
import matplotlib.pyplot as plt  

# Load and normalize features  
df = pd.read_csv('data/qawwali_features.csv', index_col='song_id')  
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
df_clustered.to_csv('data/qawwali_features_clustered.csv')  
  
print(f'Cluster distribution:')  
print(df_clustered['cluster'].value_counts().sort_index())  
EOF
```
### 2. Run clustering
```
python src/5_load_csv.py
```
### 3. Verify clustering results
```
tail -5 data/qawwali_features_clustered.csv  
python -c "  
import pandas as pd  
df = pd.read_csv('data/qawwali_features_clustered.csv', index_col='song_id')  
print(f'Cluster sizes: {df[\"cluster\"].value_counts().to_dict()}')  
print(f'Total songs: {len(df)}')  
"
```
### 4. Check dendrogram creation
```
ls -la figures/dendrogram.png
```
Files used: Generated `src/5_load_csv.py`, `data/qawwali_features.csv`\
Outputs generated:
```
data/qawwali_features_clustered.csv
figures/dendrogram.png
```
Verification: 4 clusters with reasonable size distribution, no singletons or collapsed clusters

## Phase 4B (Part 1) — Interpretive Cluster Profiling
Purpose: Generate descriptive profiles for each discovered cluster using feature statistics.

### 1. Run cluster profiling script
```
python src/6_cluster_profiling.py
```
### 2. Verify centroid table creation
```
cat data/cluster_centroids.csv
```
### 3. Check profile plots
```
ls -la figures/cluster_*_profile.png
```
### 4. Validate cluster interpretations
```
python -c "  
import pandas as pd  
centroids = pd.read_csv('data/cluster_centroids.csv', index_col='cluster')  
print('Cluster centroids (rounded):')  
print(centroids.round(2))  
print('\nFeature rankings per cluster:')  
for cluster_id in centroids.index:  
    print(f'\\nCluster {cluster_id} top features:')  
    ranked = centroids.loc[cluster_id].sort_values(ascending=False)  
    print(ranked.head(3))  
"
```
Files used: `src/6_cluster_profiling.py` \
Outputs generated:
```
data/cluster_centroids.csv
figures/cluster_*_profile.png (4 files)
```
Verification: Each cluster shows distinct feature patterns, interpretations remain audio-only and descriptive

## Phase 4B (Part 2) — PCA Visualization
Purpose: Visualize cluster separation in 2D space using PCA for exploratory analysis.

### 1. Run PCA visualization script
```
python src/7_pca_visualization.py
```
### 2. Check explained variance
```
python -c "  
import pandas as pd  
from sklearn.decomposition import PCA  
df = pd.read_csv('data/qawwali_features_clustered.csv', index_col='song_id')  
X = df.drop(columns=['cluster'])  
pca = PCA(n_components=2)  
X_pca = pca.fit_transform(X)  
print(f'PC1 explained variance: {pca.explained_variance_ratio_[0]:.3f}')  
print(f'PC2 explained variance: {pca.explained_variance_ratio_[1]:.3f}')  
print(f'Total explained variance: {pca.explained_variance_ratio_.sum():.3f}')  
"
```
### 3. Verify PCA plots
```
ls -la figures/pca_clusters*.png
```
Files used: `src/7_pca_visualization.py` \
Outputs generated:
```
figures/pca_clusters.png
figures/pca_clusters_labeled.png
```
Verification: ~60% total variance explained, clear but not perfect cluster separation (expected for unsupervised audio features)

## Phase 4B (Part 3) — Big Data Analytics Layer (Spark / Hive)
Purpose: Demonstrate scalable analytics using Spark SQL on clustered feature data.

### 1. Start Spark shell
```
pyspark
```
### 2. Inside Spark shell, load and analyze data
```
from pyspark.sql import SparkSession  
spark = SparkSession.builder.appName("QawwaliBDA").getOrCreate()  

# Load CSV data  
df = spark.read.csv(  
    "data/qawwali_features_clustered.csv",  
    header=True,  
    inferSchema=True  
)  

# Show schema and sample data  
df.printSchema()  
df.show(5)  

# Cluster size aggregation  
df.groupBy("cluster").count().show()  

# Per-cluster feature means  
df.groupBy("cluster").mean().show(truncate=False)  

# Register as SQL table  
df.createOrReplaceTempView("qawwali_clusters")  

# Spark SQL query  
spark.sql("""  
SELECT  
  cluster,  
  COUNT(*) AS num_performances,  
  AVG(tabla_energy_mean) AS avg_tabla_energy,  
  AVG(taali_activity) AS avg_taali_activity  
FROM qawwali_clusters  
GROUP BY cluster  
ORDER BY avg_tabla_energy DESC  
""").show()  

# Save as Parquet  
df.write.mode("overwrite").parquet("data/qawwali_features_clustered.parquet")  

# Verify Parquet reload  
spark.read.parquet("data/qawwali_features_clustered.parquet").show(5)
```

### 3. Exit Spark and verify outputs
```
exit  
ls -la data/qawwali_features_clustered.parquet/
```
Files used: `data/qawwali_features_clustered.csv`, Spark shell \
Outputs generated:

- `data/qawwali_features_clustered.parquet/`
- Console output showing cluster statistics

Verification: Parquet files created, data integrity maintained on reload, Spark SQL queries execute successfully.\

# Notes
- All scripts assume execution from project root directory
- The pipeline processes 34 Qawwali performances (~60 seconds each)
- Spark runs in local mode; no distributed cluster required
- Intermediate large files (songs-data.npy, tt-features.npy) are Git-ignored
- Final analytic artifacts (*.csv, *.parquet, figures/*.png) are tracked
- Each phase must complete successfully before proceeding to the next
- The entire pipeline is reproducible on macOS with Python 3.10+
