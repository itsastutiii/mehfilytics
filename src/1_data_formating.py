# simple script to directly convert the MP3s to the expected numpy format
# used in lieu of qdsb, since in the case of this implementation, the data was already in the form of .mp3 files present in data/dataset
# if data extraction ya kuch is required, PLEASE use qdsb and the OLD WAY OF IMPLEMENTATION (0_preprocessing.py)

import librosa  
import numpy as np  
from pathlib import Path  

data_dir = Path('data/dataset')  
output_data = {}  
 
for mp3_file in data_dir.glob('*.mp3'):  
    song_id = mp3_file.stem  
    audio, sr = librosa.load(mp3_file, sr=44100, mono=True)  
    output_data[song_id] = audio  
  
output_data['rate'] = 44100  
np.save('data/dataset/songs-data.npy', output_data) # type: ignore

# the initial pipeline that has 4 extra files (now deleted or moved to preprocessing.py) were problmeatic
# i think schema ya kahi there was a hardcoded behaviour that was causing the url/path to think that the name of each mp3 file will be of format "name".mp3 instead of fid.mp3