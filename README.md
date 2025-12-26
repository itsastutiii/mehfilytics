# Mehfilytics -- Performance-Based Acoustic Analysis of Qawwali Music
Based on the dataset and architecture of the project forked. 

🌙 This project explores whether distinct performance-based acoustic regimes emerge within Qawwali music when analyzed using audio-only, unsupervised methods, and organized through a scalable Big Data Analytics (BDA) pipeline.\
It is not (yet!) a project about lyrics, devotion, or religious meaning, primarily dealing with how Qawwali is performed (rhythm, percussion, vocal activity, and intensity) as heard in sound.

# IO Description
## Inputs
### 1. Audio Dataset
```
Curated subset of the QawwalRang dataset
34 validated audio clips
~60 seconds each
Mono, resampled to 44.1 kHz
```
The dataset was intentionally frozen after validation to ensure signal correctness before analytics.
### 2. Derived Structured Data
Mid-level acoustic feature vectors per clip. \
Features represent:
- Tabla energy and rhythmic activity
- Taali / vocal transient density
- Spectral characteristics (MFCC-based)

Each clip is represented as a single interpretable feature vector
### 3. Metadata
- Song ID
- Artist
- Source information
- Duration
Metadata is not used for clustering, only for post-hoc analysis (e.g., artist–cluster relationships).

## Outputs
This project produces analytic outputs, not models or classifiers.
### 1. Primary Artifacts - Cluster Assignments
Each audio clip is assigned a cluster ID via unsupervised hierarchical clustering.\
Primary output file:
```
data/qawwali_features_clustered.csv
data/qawwali_features_clustered.parquet
````
Each row 
```song_id × acoustic_features × cluster_id```

### 2. Analytic Artifacts - Cluster Profiles
For each cluster, the following are computed:
- Mean tabla energy
- Mean rhythmic activity
- Mean taali activity
- Spectral variation characteristics\

Artifacts:
- Cluster centroid tables
- Cluster feature profile plots
- PCA visualization (for interpretability only)

### 3. Interpretive Analysis
Clusters are interpreted as performance regimes, for example:
- Percussion-dominant, rhythmically dynamic passages
- Vocal / taali-forward call-response sections
- Restrained or transitional performances
- Steady-state rhythmic grounding sections
These interpretations are numerically justified using cluster-wise statistics and are not semantic or genre labels.

### 4. System Validation
- Spark ingestion
- SparkSQL / Hive-style analytics
- Parquet storage

### Final Output
Discovery of multiple performance-based acoustic regimes in Qawwali music using audio-only, unsupervised analysis, with results represented as structured analytic artifacts compatible with Spark and Hive-based Big Data workflows.

# Architecture 
```
Audio (QawwalRang)
        ↓
Feature Extraction (Python / librosa)
        ↓
Structured Feature Table (CSV / Parquet)
        ↓
Spark (local / cluster mode)
        ↓
SparkSQL / Hive-style Queries
        ↓
Analytic Results + Visualizations + Interpretation
```

## Design Decision of BDA
Raw audio signal processing is not performed inside Hadoop/Spark, as distributed frameworks are unsuitable for low-level signal analysis. Instead, signal-level processing is completed first (best practice in MIR).\
Spark is used for:
- structured aggregation
- clustering validation
- SQL-style analytics
- scalable querying

## Spark / Hive Layer
The structured feature table is ingested into Apache Spark (local mode) and analyzed using:
- Distributed DataFrame operations
- groupBy, avg, and aggregation queries
- SparkSQL (Hive-style temporary views)
- Columnar storage using Parquet


Example analyses include:
- Cluster size distributions
- Per-cluster acoustic statistics
- Artist–cluster aggregation queries
Although the dataset size is modest, the pipeline mirrors a scalable Big Data workflow and can be deployed to HDFS without code changes.

# Progress 
## Phase 1 -- Forking, Dataset Reconstruction & Curation.
Audio File Path:
```
metadata/original_qawali_metadata_modified.json
```

34 is fine because this isn't a supervised classification task. It mostly deals with exploratory clustering, mid-level acoustic features & performance-centric analysis. Itne clips are common in MIR exploratory studies, statistically stable for low-variance features, and strengthened by artist diversity and intra-artist variation. 

## Phase 2 -- Signal-Level Feature Extraction 
Before Spark, Hive, or HDFS make sense, audio must be normalized and fixed up.
```
Audio normalization using qdetect.py.
Source separation into: tabla (percussion beats) and taali (vocal/clapping proxy).
Feature extraction: Tabla → CQT (high frequency resolution), Taali → MFCC (timbral + transient proxy).
```
Command:
```
python src/qdetect.py data/qawwali_audio_clean --reload --extract
```
Output:
```
songs-data.npy
features/tt-features.npy
```
Each clip produced time–frequency matrices (not vectors).

## Phase 3 -- Feature Reduction & Validation
Time-frequency matrices are not clusterable, toh theyre reduced into interpretable, per-song features, capturing:
* percussion intensity
* rhythmic dynamism
* taali activity
* temporal burstiness

Final Feature Table Dim
```
34 songs × 8 features
```
Exported as:
```
data/qawwali_features.csv
```
Validation Performed:
- Range inspection
- Z-score normalization
- Correlation analysis

Tabla and taali form partially independent performance axes. Correlations are interpretable, not pathological.

## Phase 4
### Unsupervised Discovery (Hierarchical Clustering)
Method
```
Distance: Euclidean (on normalized features)
Clustering: Hierarchical (Ward’s linkage)
Visualization: Dendrogram
```
Result
```
Clear hierarchical structure
Natural cut at 4 clusters
No forced symmetry, no fragmentation
(will insert result screenshot images here baad me)
```
This directly answers the core research question. i.e. yes, distinct performance-based acoustic regimes emerge within Qawwali.

### Interpretive Cluster Profiling
Added cluster profiling and PCA visualisation.\
Clusters were interpreted without semantic labels, revealing regimes such as:
```
percussion-dominant, highly dynamic performances
taali / call–response-heavy sections
restrained, low-activity passages
steady-state rhythmic grounding
```
All interpretations are:
```
audio-only
feature-backed
descriptive, not declarative
```
NOTE: No claim is made for classification like “Cluster X = ghazal / hamd / naat”.\
Just “Cluster X exhibits high rhythmic density and transient activity.”\
That distinction is deliberate. 

### Big Data Analytics Layer
Spark / Hive Analytics Layer (Operationalizing the Results in a BDA Stack)\
Goal: Demonstrate that the structured acoustic analytics output can be ingested, queried, and analyzed using distributed Big Data tools (Spark / Hive).

**Gyaan:**
Planned BDA integration includes:
- Storing audio, metadata, and features in HDFS
- Loading feature tables into Spark
- Running SparkSQL / Hive queries
- Demonstrating pipeline scalability


Why this for BDA lab?
This isnt "big data analytics" in the naïve sense of petabytes, but it is in the correct academic sense, because:
- heterogeneous data types are handled
- a full analytics pipeline is implemented
- tools scale beyond this dataset
- method, not volume, is the focus
While the dataset is modest in size, the pipeline is designed to scale to larger music corpora.
