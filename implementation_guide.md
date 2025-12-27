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

### 1. Extraction of Data
The .mp3 files that needed for running this project are in a compressed form under /data. the name of the file will be /dataset. To find it, unzip the compressed folder, retrieve /dataset, and drag-drop it to /data. If you prefer extracting the authentic way, follow the fork to the original repository and use it. I could only extract 34 files in that manner, which is why I did it this way rn. 

### 2. Verify audio files
```
ls -la data/dataset/*.mp3 | wc -l  
# Should show 72 files
```
### 3. Check audio format consistency
```
python -c "  
import librosa  
import glob  
for f in glob.glob('data/dataset/*.mp3')[:5]:  
    y, sr = librosa.load(f, sr=None)  
    print(f'{f}: {sr}Hz, mono={len(y.shape)==1}, duration={len(y)/sr:.2f}s')  
"
```
Expected output sample:
```
data/dataset/q059.mp3: 44100Hz, mono=True, duration=60.00s
...
```

### 4. Dataset Construction (Numerical Representation)
Once verified, the raw audio files are converted into a unified numerical dataset using:
```
python src/1_data_formating.py
```
If you face a warning, that's fine, verify the action using the snippet below:
```
python -c "  
import numpy as np
data = np.load('data/dataset/songs-data.npy', allow_pickle=True).item()
print(f'Songs loaded: {len([k for k in data.keys() if k != \"rate\"])}')
print(f'Sample rate: {data[\"rate\"]}Hz')
"
```
Expected Output:
```
Songs loaded: 72
Sample rate: 44100Hz
```

This step:
- Loads all .mp3 files from `data/dataset/`
- Normalizes sampling rate
- Stores audio waveforms in a single serialized NumPy object


Outputs of Phase 1:
```
data/dataset/
├── q001.mp3
├── q002.mp3
├── ...
├── q072.mp3
└── songs-data.npy
```
Verification summary:
- 72 audio files
- 44.1 kHz sampling rate
- Mono channel
- ~60 seconds per clip
- Successfully serialized into songs-data.npy

## Phase 2 — Signal-Level Feature Extraction
Purpose: Extract tabla (CQT) and taali (MFCC) features from audio using source separation.

### 1. Feature Extraction and Source Separation
Signal-level features are extracted by running the detection and extraction pipeline:
```
python src/qdetect.py data/dataset --reload --extract
```
Expected Output Sample:
```
...
2025-12-27 12:56:45,082 - __main__ - 
Classification starting for song: q040
2025-12-27 12:56:45,089 - __main__ - q040 categorized as Qawali after detecting tabla and taali
2025-12-27 12:56:45,089 - __main__ - 
Classification starting for song: q054
2025-12-27 12:56:45,094 - __main__ - Tabla not detected, model calculated pitch-power mean=71.62558025856795 and std-dev 15.92587728804688
2025-12-27 12:56:45,094 - __main__ - 
--------------------Classification Results----------------------------

2025-12-27 12:56:45,094 - __main__ - Total=72 non-Qawalis=18 TablaTaali=48 Tabla=0 Taali=6
```
This step performs the following operations for each audio clip:
* Separates tabla and taali sources using NMF-based decomposition
* Extracts:
  * Tabla features as Constant-Q Transform (CQT) representations
  * Taali features as Mel-Frequency Cepstral Coefficients (MFCCs)
* Stores time–frequency matrices for downstream aggregation


### 2. Verification of Feature Files
The extracted features are saved as a serialized NumPy object:
```
data/dataset/features/tt-features.npy
```
Verify file creation:
```
ls -la data/dataset/features/
```

### 3. Feature Matrix Integrity Checks (opt)
To inspect the structure of extracted features:
```
python -c "
import numpy as np

features = np.load(
    'data/dataset/features/tt-features.npy',
    allow_pickle=True
).item()

for key, mat in list(features.items())[:4]:
    print(f'{key}: shape={mat.shape}, type={type(mat)}')
"
```
### 4. Validation of Time–Frequency Representations (opt)
Confirm expected dimensionality for each feature type:
```
python -c "
import numpy as np

features = np.load(
    'data/dataset/features/tt-features.npy',
    allow_pickle=True
).item()

tabla_keys = [k for k in features.keys() if 'tabla' in k]
taali_keys = [k for k in features.keys() if 'taali' in k]

print(f'Tabla features: {len(tabla_keys)} files')
print(f'Taali features: {len(taali_keys)} files')
print(f'Tabla CQT bins: {features[tabla_keys[0]].shape[0]} (expected: 84)')
print(f'Taali MFCC coeffs: {features[taali_keys[0]].shape[0]} (expected: 13)')
"
```

Outputs of Phase 2: 
```
data/dataset/features/
└── tt-features.npy

```
Verification summary:
- Tabla features: 84 CQT frequency bins
- Taali features: 13 MFCC coefficients
- Features stored as time–frequency matrices for all processed songs


## Phase 3 — Feature Reduction & Validation
Purpose: To convert high-dimensional time–frequency representations into a compact, interpretable per-song feature vector suitable for clustering and statistical analysis.

### 1. Feature Reduction
The signal-level features extracted in Phase 2 (CQT for tabla, MFCC for taali) are aggregated into scalar descriptors using:
```
python src/3_reduce_features.py
```
Output Sample:
```

         tabla_energy_mean  tabla_energy_var  ...  taali_activity  taali_burstiness
song_id                                       ...                                  
q059              0.008847          0.001225  ...        0.209946          0.000834
q071              0.009292          0.000912  ...        0.000000          0.000458
q065              0.023772          0.007164  ...        0.469621          0.001975
q064              0.014795          0.002556  ...        0.139899          0.002016
q070              0.018744          0.004673  ...        0.000000          0.000967

[5 rows x 8 columns]
(72, 8)
```
This step:
- Aggregates time–frequency matrices across time
- Computes energy statistics, activity measures, and distributional descriptors
- Produces a fixed-length feature vector for each song


### 2. Verify CSV output structure
```
head -5 data/dataset_72_qawwali_features.csv  
wc -l data/dataset_72_qawwali_features.csv 
```
### 3. Feature Dimensions and Ranges (Optional)
```
python -c "
import pandas as pd

df = pd.read_csv('data/dataset_72_qawwali_features.csv', index_col='song_id')
print(df.describe().loc[['min', 'mean', 'max']])
"
```
This confirms:
- Non-negative energy features
- Reasonable dynamic ranges
- No degenerate (constant) columns
Outputs of Phase 3
```
data/
└── dataset_72_qawwali_features.csv
```
Verification summary:
- Fixed-length representation per song
- 8 interpretable acoustic features
- Suitable for clustering, PCA, and profiling


## Phase 4 — Unsupervised Pattern Discovery & Visualization
This phase applies unsupervised learning techniques to the reduced feature matrix obtained in Phase 3 in order to (i) discover latent performance groupings and (ii) visualize structure in a lower-dimensional space.

### Phase 4A — Unsupervised Discovery (Hierarchical Clustering)
Purpose: 
To identify natural groupings of Qawwali performances based purely on acoustic feature similarity, without using labels or metadata.

#### 1. Hierarchical Clustering
Hierarchical clustering is performed on the standardized feature matrix using Ward’s linkage:
```
python src/4_load_csv.py
```
Sample console output:
```
Cluster distribution:
1    10
2    14
3    23
4    25
```
This script:
- Loads the reduced feature CSV
- Standardizes features
- Computes pairwise Euclidean distances
- Applies Ward hierarchical clustering
- Cuts the dendrogram at 4 clusters


#### 2. Verification of Clustering Results
```
python -c "
import pandas as pd

df = pd.read_csv('data/dataset_72_features_clustered.csv', index_col='song_id')
print(f'Cluster sizes: {df[\"cluster\"].value_counts().to_dict()}')
print(f'Total songs: {len(df)}')
"
```

Outputs (Phase 4A):
```
data/
└── dataset_72_features_clustered.csv

figures/
└── dendrogram.png
```
Verification summary:
- 4 clusters of reasonable and balanced size
- No collapsed clusters
- Clustering driven purely by acoustic features


#### 3. Interpretive Cluster Profiling
Run cluster profiling script
```
python src/5_cluster_profiling.py
```
Verify centroid table creation
```
cat data/cluster_centroids.csv
```
Outputs (Cluster Profiling)
```
data/
└── cluster_centroids.csv

figures/cluster_*_profile.png (4 files)
```
Verification: Each cluster shows distinct feature patterns, interpretations remain audio-only and descriptive

### Phase 4B — PCA-Based Visualization
Purpose: To visualize the distribution of songs and clusters in a low-dimensional space for exploratory analysis.

#### 1. PCA Projection
```
python src/6_pca_visualization.py
```
Console output:
```
Explained variance ratio: [0.341, 0.229]
Total explained variance: 0.570
```
This indicates that the first two principal components capture ~57% of total variance.

#### 2. PCA Plots
Generated plots:
```
figures/dataset_72_pca_clusters.png
figures/dataset_72_pca_clusters_labeled.png
```
Outputs (Phase 4B):
```
figures/
├── dataset_72_pca_clusters.png
└── dataset_72_pca_clusters_labeled.png
```
Verification summary:
- Moderate but meaningful cluster separation
- Partial overlap expected due to unsupervised setting
- PCA used strictly for visualization, not clustering


## Phase 5 — Big Data Analytics Layer (Spark SQL)
Purpose:  To demonstrate scalable, distributed analytics on the clustered Qawwali feature dataset using Apache Spark.
This phase validates that the reduced and clustered features can be seamlessly integrated into a Big Data processing framework, even though the current dataset size is modest.


⚠️ Note: This phase focuses on pipeline compatibility and scalability, not performance benchmarking.


### 1. Launch Spark Shell
Activate the virtual environment and start PySpark:
```
pyspark
```
A local Spark session is initialized (master = local[*]).

### 2. Load Clustered Feature Data in Spark
Inside the Spark shell:
```
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("QawwaliBDA").getOrCreate()

df = spark.read.csv(
    "data/dataset_72_features_clustered.csv",
    header=True,
    inferSchema=True
)
```
### 3. Inspect Schema and Sample Records
```
df.printSchema()
df.show(5)
```
Expected schema:
```
song_id: string
tabla_energy_mean: double
tabla_energy_var: double
tabla_lowband_ratio: double
tabla_activity: double
taali_mfcc_mean: double
taali_mfcc_var: double
taali_activity: double
taali_burstiness: double
cluster: integer
```
### 4. Cluster-Level Aggregations (Spark DataFrame API)
Cluster size distribution:
```
df.groupBy("cluster").count().show()
```
Sample output:
```
cluster | count
----------------
1       | 10
2       | 14
3       | 23
4       | 25
```
Per-cluster mean feature values:
```
df.groupBy("cluster").mean().show(truncate=False)
```
This reproduces cluster centroids previously computed in Phase 4, now using distributed execution.
### 5. Spark SQL Analysis
Register the DataFrame as a temporary SQL table:
```
df.createOrReplaceTempView("qawwali_clusters")
```
Run a SQL query to compare clusters:
```
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
```
This demonstrates:
* SQL-based analytics on acoustic feature data
* Interoperability between DataFrame and SQL APIs


### 6. Persist Data in Columnar Format (Parquet)
To simulate scalable storage and downstream analytics:
```
df.write.mode("overwrite").parquet(
    "data/dataset_72_features_clustered.parquet"
)
```
Verify successful reload:
```
spark.read.parquet(
    "data/dataset_72_features_clustered.parquet"
).show(5)
```
### 7. Exit Spark and Verify Output
Exit the Spark shell:
```
exit()
```
Verify Parquet files:
```
ls -la data/dataset_72_features_clustered.parquet/
```
Expected contents:
```
_SUCCESS
part-00000-*.snappy.parquet
```
Outputs of Phase 5:
```
data/
└── dataset_72_features_clustered.parquet/
    ├── part-00000-*.snappy.parquet
    └── _SUCCESS
```
Verification Summary:
- Clustered feature data successfully loaded into Spark
- Aggregations and SQL queries executed correctly
- Parquet persistence and reload verified
- Pipeline confirmed to be compatible with Big Data analytics frameworks


## Notes
- All scripts assume execution from project root directory
- The pipeline processes 34 Qawwali performances (~60 seconds each)
- Spark runs in local mode; no distributed cluster required
- Intermediate large files (songs-data.npy, tt-features.npy) are Git-ignored
- Final analytic artifacts (*.csv, *.parquet, figures/*.png) are tracked
- Each phase must complete successfully before proceeding to the next
- The entire pipeline is reproducible on macOS with Python 3.10+
