# tools/check_wavs.py
import os, glob
import librosa
import numpy as np

def rms(y):
    return float(np.sqrt(np.mean(y**2))) if y.size else 0.0

for p in glob.glob("dataset_synthetic/**/*.wav", recursive=True):
    try:
        y, sr = librosa.load(p, sr=16000)
        print(p, "dur", len(y)/sr, "s", "rms", rms(y))
    except Exception as e:
        print("ERROR", p, e)