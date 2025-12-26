import numpy as np
import pandas as pd

FEATURE_PATH = "data/qawwali_audio_clean/features/tt-features.npy"
OUT_CSV = "data/qawwali_features.csv"

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


# OUTPUT

#          tabla_energy_mean  tabla_energy_var  ...  taali_activity  taali_burstiness
# song_id                                       ...                                  
# q012              0.010104          0.000702  ...        0.028157          0.000599
# q006              0.016255          0.005761  ...        0.742912          0.001857
# q007              0.019860          0.007037  ...        0.000000          0.001738
# q013              0.029122          0.015438  ...        0.618674          0.001513
# q039              0.021258          0.010315  ...        0.000000          0.001483

# [5 rows x 8 columns]
# (34, 8)