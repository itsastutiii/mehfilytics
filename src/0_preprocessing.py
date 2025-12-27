# all these steps are needed when the audio files need to be extracted and processed.  
# DONT RUN THIS FILE, IT IS ONLY HERE FOR CONTEXT

# 1_fix_metadata.py
# modifying the download: false thing so we can extract mp3 files from youtube links 
# acts on the original json file from the repo
import json
from pathlib import Path

INPUT = Path("metadata/qawali_metadata.json")
OUTPUT = Path("metadata/qawali_metadata_fixed.json")

with open(INPUT, "r") as f:
    data = json.load(f)

fixed = 0
name_fixed = 0

for q in data["qawalian"]:
    # Enable download for YouTube links
    if "youtube.com" in q.get("url", "") or "youtu.be" in q.get("url", ""):
        if not q.get("download", False):
            q["download"] = True
            fixed += 1

    # Fix accidental ".mp3.mp3" situations
    name = q.get("name", "")
    if name.endswith(".mp3"):
        q["name"] = name.replace(".mp3", "")
        name_fixed += 1

with open(OUTPUT, "w") as f:
    json.dump(data, f, indent=2)

print(f"✅ Updated download flag for {fixed} YouTube entries")
print(f"🧹 Cleaned {name_fixed} file names")
print(f"📄 Written to {OUTPUT}")

#2_clean_metadata.py
# used to create the new json file of metadata of all the mp3 files 
# that were successfully retreived from the extraction attempt.
import json

# Paths
ORIGINAL_JSON = "metadata/qawali_metadata_fixed.json"
VALID_FIDS_FILE = "valid_fids.txt"
OUTPUT_JSON = "metadata/metadata_of_retreived.json"

# Load valid fids
with open(VALID_FIDS_FILE) as f:
    valid_fids = set(line.strip() for line in f)

# Load original metadata
with open(ORIGINAL_JSON) as f:
    data = json.load(f)

clean_entries = []

for entry in data["qawalian"]:
    fid = entry.get("fid")
    if fid in valid_fids:
        clean_entry = {
            "fid": fid,
            "name": entry.get("name"),
            "artist": entry.get("artist"),
            "download": False,
            "start": entry.get("start", 0),
            "duration": entry.get("duration", 60),
            "thaat": entry.get("thaat"),
            "raag": entry.get("raag"),
            "taal": entry.get("taal")
        }
        clean_entries.append(clean_entry)

clean_data = {
    "qawalian": clean_entries
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(clean_data, f, indent=2)

print(f"✅ Clean metadata written to {OUTPUT_JSON}")
print(f"🎵 Total clips: {len(clean_entries)}")

#3_quick_test.py
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