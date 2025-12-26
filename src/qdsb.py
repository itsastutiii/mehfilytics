# Copyright (c) 2022 Faheem Sheikh

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

# qawali dataset builder(qdsb)
# Constructs reference qawali dataset from associated metadata information
# It reads/downloads original songs, then extracts short duration audio from them,
# finally writing back to a user-defined data-set location in a compressed format.

import argparse
from yt_dlp import YoutubeDL
from ffmpy import FFmpeg as ffm
import gdown as gd
import json
import librosa as rosa
import logging
from pathlib import Path
import os
import subprocess
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('TkAgg')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


class QawalRang:
    SampleRate = 44100
    InFormat = '.mp3'
    InterFormat = '.wav'

    QawalRangSources = 'sources.png'
    QawalRangArtists = 'artist.png'
    QawalRangProps = 'props.png'
    WebSize = 6

    def __init__(self, target_path, metadata_file, offline_location=None):
        logger.info("Qawali dataset construction started")
        self.m_target = Path(target_path)
        self.m_local = offline_location
        if offline_location is None:
            logger.warning("No offline location specified, all songs will be downloaded")
        with open(metadata_file) as j_file:
            self.m_qmap = json.load(j_file)

    def __download(self, song):
        if song['name'] is None or song['url'] is None:
            logger.error("Invalid metadata entry, skipping")
            return

        logger.info("Download requested for %s from %s", song['name'], song['url'])
        out_song = self.m_target / song['fid']

        if out_song.with_suffix(self.InFormat).exists():
            logger.info("Song %s already exists, skipping", out_song.name)
            return

        if "youtube" in song['url']:
            ydl_params = {
                'format': 'bestaudio/best',
                'outtmpl': str(out_song.with_suffix(self.InFormat)),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '128',
                }],
                'postprocessor_args': [
                    '-ss', str(song['start']),
                    '-t', str(song['duration'])
                ],
                'noplaylist': True,
                'logger': logger,
                'progress_hooks': [QawalRang.download_progress],
            }

            try:
                with YoutubeDL(ydl_params) as ydl:
                    ydl.download([song['url']])
            except Exception as e:
                logger.error("Failed to download from YouTube: %s", e)

        elif "google" in song['url']:
            try:
                gd.download(
                    song['url'],
                    str(out_song.with_suffix(self.InFormat)),
                    quiet=True,
                    proxy=None
                )
            except Exception as e:
                logger.error("Failed to download from Google Drive: %s", e)

        else:
            logger.error("Unsupported URL: %s", song['url'])

    def __write(self, song):
        song_location = Path(self.m_local) / song['name']
        in_song = song_location.with_suffix(self.InFormat)

        if not in_song.exists():
            logger.error("%s not found locally, skipping", in_song)
            return

        out_song = self.m_target / song['fid']
        if out_song.with_suffix(self.InFormat).exists():
            return

        song_data, sr = rosa.load(
            path=in_song,
            sr=self.SampleRate,
            mono=True,
            offset=float(song['start']),
            duration=float(song['duration']),
            dtype='float32'
        )

        rosa.output.write_wav(
            str(out_song.with_suffix(self.InterFormat)),
            y=song_data,
            sr=self.SampleRate
        )

        try:
            compressor = ffm(
                inputs={str(out_song.with_suffix(self.InterFormat)): None},
                outputs={str(out_song.with_suffix(self.InFormat)): None}
            )
            compressor.run(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            logger.error("Compression failed: %s", e)

        os.remove(str(out_song.with_suffix(self.InterFormat)))

    def make(self):
        logger.info("Making dataset...")
        for qawali in self.m_qmap['qawalian']:
            if qawali.get('download', False):
                self.__download(qawali)
            elif self.m_local is not None:
                self.__write(qawali)
            else:
                logger.warning("No source available for %s, skipping", qawali['name'])

    @staticmethod
    def download_progress(prog):
        if prog.get('status') == 'downloading':
            logger.info("Download progress: %s", prog.get('_percent_str', ''))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qawwali dataset builder")
    parser.add_argument("datapath")
    parser.add_argument("metadata")
    parser.add_argument("--opath", dest="offline_path")
    parser.add_argument("--info", action="store_true")

    args = parser.parse_args()

    d_path = Path(args.datapath)
    m_path = Path(args.metadata)

    if not d_path.exists() or not m_path.exists():
        raise SystemExit("Invalid datapath or metadata path")

    o_path = Path(args.offline_path) if args.offline_path else None

    qds = QawalRang(d_path, m_path, o_path)

    if args.info:
        qds.info()
    else:
        qds.make()
