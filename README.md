# Mehfilytics -- Performance-Based Acoustic Analysis of Qawwali Music

🌙 Mehfilytics investigates whether distinct performance based acoustic regimes emerge within Qawwali music, using audio-only, unsupervised analysis and a Big Data Analytics style pipeline.


This project deliberately studies how Qawwali is performed rhythm, percussion, vocal activity, and intensity; not lyrics, devotion, or religious semantics(yet!).


All interpretations are grounded strictly in acoustic evidence.


## Objective
To determine whether unsupervised clustering of mid-level acoustic features can reveal recurring performance regimes in Qawwali music, and to operationalize the results within a Spark-compatible analytics pipeline.


**Key constraints**:
- Audio-only (no lyrics or semantic metadata)
- Unsupervised (no labels, no classifiers)
- Interpretive but non-declarative (descriptive regimes, not genre tags)

## Data Overview
### Audio
- Curated subset of the QawwalRang dataset
- 34 validated audio clips
- ~60 seconds each
- Mono, 44.1 kHz
- Dataset intentionally frozen post-validation for signal correctness\
The dataset size is appropriate for exploratory MIR clustering and aligns with common practice for mid-level acoustic studies emphasizing interpretability over scale.

### Metadata
- Song ID
- Artist
- Source information
- Duration
Metadata is not used for clustering, only for optional post-hoc analysis.

## Feature Representation
Each audio clip is reduced to a **single interpretable feature vector** capturing:
- Tabla energy and rhythmic activity
- Taali / vocal transient density
- Spectral and temporal variation characteristics\
Final feature table:
```
34 songs × 8 acoustic features
```
## Outputs
This project produces analytic artifacts, not predictive models.
### 1. Cluster Assignments
Each performance is assigned a cluster via unsupervised hierarchical clustering.\
Primary outputs:
- `data/qawwali_features_clustered.csv`
- `data/qawwali_features_clustered.parquet`

Structure:
`song_id × acoustic_features × cluster_id`

### 2. Cluster Profiles & Visualizations
For each cluster:
- Feature centroids
- Relative feature importance
- Per-cluster profile plots
- PCA visualization (interpretive only)\
These artifacts support numerically justified interpretation of performance regimes.

### 3. Interpretive Findings
Clusters correspond to performance regimes, such as:
- Percussion-dominant, rhythmically dynamic passages
- Taali / call–response–heavy sections
- Restrained or low-activity passages
- Steady-state rhythmic grounding sections\
⚠️ Important:\
These are descriptive acoustic regimes, not semantic or genre labels.\
No claims are made such as “Cluster X = hamd / naat / ghazal.”

### 4. Big Data Analytics Validation
The structured feature outputs are:
- Ingested into Apache Spark (local mode)
- Queried using SparkSQL / Hive-style analytics
- Stored in columnar Parquet format
This demonstrates that the pipeline is:
- Scalable
- Framework-compatible
- Deployable to HDFS without code changes

## System Architecture
```
Raw Audio (QawwalRang)
        ↓
Signal-Level Feature Extraction (Python / librosa)
        ↓
Structured Feature Table (CSV / Parquet)
        ↓
Spark (local or cluster mode)
        ↓
SparkSQL / Hive-style Analytics
        ↓
Analytic Results + Visualizations + Interpretation
```

## Design Rationale (BDA)
Low-level signal processing is intentionally performed outside Spark/Hadoop, in line with best practices in Music Information Retrieval.

Spark is used for:
- Structured aggregation
- Cluster validation
- SQL-style analytics
- Scalable querying and storage\
The focus is methodological scalability, not dataset size.

## Project Status
- Dataset reconstruction & validation ✔
- Feature extraction & reduction ✔
- Unsupervised clustering ✔
- Interpretive profiling & PCA ✔
- Spark / Parquet analytics ✔


All implementation details, commands, and verification steps are documented in `implementation_guide.md`.