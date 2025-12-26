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
