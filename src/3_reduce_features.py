# converts time-frequency matrices to 8 scalar features per song 
# outputting qawwali_features.csv

import numpy as np
import pandas as pd

# Note to self -- make this dynamic
FEATURE_PATH = "data/dataset/features/tt-features.npy"
OUT_CSV = "data/dataset_72_qawwali_features.csv"

data = np.load(FEATURE_PATH, allow_pickle=True).item()

songs = {}

def activity_ratio(mat, threshold=0.1):
    frame_energy = mat.mean(axis=0)
    return (frame_energy > threshold * frame_energy.max()).mean()

def burstiness(mat):
    frame_energy = mat.mean(axis=0)
    return np.std(np.diff(frame_energy))

for key, mat in data.items():
    song_id, part = key.split(".")

    if song_id not in songs:
        songs[song_id] = {}

    if part == "tabla":
        songs[song_id]["tabla_energy_mean"] = mat.mean()
        songs[song_id]["tabla_energy_var"] = mat.var()
        songs[song_id]["tabla_lowband_ratio"] = mat[:20].sum() / mat.sum()
        songs[song_id]["tabla_activity"] = activity_ratio(mat)

    elif part == "taali":
        songs[song_id]["taali_mfcc_mean"] = mat.mean()
        songs[song_id]["taali_mfcc_var"] = mat.var()
        songs[song_id]["taali_activity"] = activity_ratio(mat)
        songs[song_id]["taali_burstiness"] = burstiness(mat)

df = pd.DataFrame.from_dict(songs, orient="index")
df.index.name = "song_id"

print(df.head())
print(df.shape)

df.to_csv(OUT_CSV)
