import os
from functools import partial
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

# Constants for audio processing
FPS = 30
HOP_LENGTH = 512
SAMPLE_RATE = FPS * HOP_LENGTH

def extract_audio_features(file_path, skip_completed=True, dest_dir="baseline_feats"):
    """Extract audio features from a file and save them to a directory."""
    os.makedirs(dest_dir, exist_ok=True)
    audio_name = Path(file_path).stem
    save_path = os.path.join(dest_dir, audio_name + ".npy")

    if os.path.exists(save_path) and skip_completed:
        return

    try:
        data, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return

    envelope = librosa.onset.onset_strength(y=data, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=data, sr=SAMPLE_RATE, n_mfcc=20).T
    chroma = librosa.feature.chroma_cens(y=data, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, n_chroma=12).T

    peak_indices = librosa.onset.onset_detect(onset_envelope=envelope, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    peak_onehot = np.zeros_like(envelope, dtype=np.float32)
    peak_onehot[peak_indices] = 1.0

    # start_bpm = librosa.beat.tempo(y=data, sr=SAMPLE_RATE)[0]
    start_bpm = librosa.beat.tempo(y=librosa.load(file_path)[0])[0]

    tempo, beat_indices = librosa.beat.beat_track(
        onset_envelope=envelope,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        start_bpm=start_bpm,
        tightness=100
    )
    beat_onehot = np.zeros_like(envelope, dtype=np.float32)
    beat_onehot[beat_indices] = 1.0

    features = np.concatenate(
        [envelope[:, None], mfcc, chroma, peak_onehot[:, None], beat_onehot[:, None]],
        axis=-1
    )

    # Ensure features have a consistent shape
    if features.shape[0] > 5 * FPS:
        features = features[:5 * FPS]

    # np.save(save_path, features)
    return features, save_path

def process_audio_directory(src_dir, dest_dir):
    """Process all audio files in the directory."""
    file_paths = sorted(Path(src_dir).glob("*"))
    extract_features = partial(extract_audio_features, skip_completed=False, dest_dir=dest_dir)
    for file_path in tqdm(file_paths):
        feats, path = extract_features(file_path)
        np.save(path,feats)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract audio features from PopDanceSet.")
    parser.add_argument("--src", required=True, help="Source directory containing audio files")
    parser.add_argument("--dest", required=True, help="Destination directory for extracted features")
    args = parser.parse_args()

    process_audio_directory(args.src, args.dest)
