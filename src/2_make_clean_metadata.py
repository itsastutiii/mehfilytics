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
