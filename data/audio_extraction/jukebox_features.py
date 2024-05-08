import os
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import jukemirlib
import numpy as np
from tqdm import tqdm

FPS = 30
LAYER = 66

def extract_audio_features(file_path: str, skip_completed: bool = True, dest_dir: str = "juke_feats") -> Optional[Tuple[np.ndarray, str]]:
# def extract_audio_features(file_path: str, skip_completed: bool = True, dest_dir: str = "juke_feats"):
    """
    Extracts audio features from a given file and saves them as a .npy file.

    Parameters:
    file_path (str): Path to the audio file.
    skip_completed (bool): If True, skips extraction if features already exist.
    dest_dir (str): Directory to save the extracted features.

    Returns:
    Tuple[np.ndarray, str]: Tuple containing the features and the save path, or None if skipped.
    """
    os.makedirs(dest_dir, exist_ok=True)
    audio_name = Path(file_path).stem
    save_path = os.path.join(dest_dir, f"{audio_name}.npy")

    if os.path.exists(save_path) and skip_completed:
        return None

    try:
        audio = jukemirlib.load_audio(file_path)
        features = jukemirlib.extract(audio, layers=[LAYER], downsample_target_rate=FPS)
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return None

    # np.save(save_path, features[LAYER])
    return features[LAYER], save_path

def process_audio_directory(src_dir: str, dest_dir: str):
    """
    Processes all audio files in the given directory to extract features.

    Parameters:
    src_dir (str): Source directory containing audio files.
    dest_dir (str): Destination directory for extracted features.
    """
    file_paths = sorted(Path(src_dir).glob("*"))
    extract_func = partial(extract_audio_features, skip_completed=False, dest_dir=dest_dir)
    for file_path in tqdm(file_paths):
        feats, path = extract_func(file_path)
        np.save(path,feats)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract audio features from files.")
    parser.add_argument("--src", required=True, help="Source path to the audio files")
    parser.add_argument("--dest", required=True, help="Destination path for audio features")
    args = parser.parse_args()

    process_audio_directory(args.src, args.dest)

