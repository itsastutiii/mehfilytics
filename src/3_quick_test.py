import numpy as np

data = np.load(
    "data/qawwali_audio_clean/features/tt-features.npy",
    allow_pickle=True
).item()

for k, v in data.items():
    if k.endswith(".taali"):
        print(k, v.shape)
        break

# output: q012.taali (13, 10335)