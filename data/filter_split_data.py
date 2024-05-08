import glob
import os
import pickle
import shutil
from pathlib import Path
import numpy as np

def file_to_list(filepath):
    with open(filepath, "r") as file:
        lines = file.readlines()
    return [line.strip() for line in lines if line.strip()]

def prepare_directories(base_dir, subfolders):
    for folder in subfolders:
        Path(base_dir / folder).mkdir(parents=True, exist_ok=True)

def copy_and_process_files(dataset_path, split_name, split_list):
    base_path = Path(dataset_path)
    motion_dir = Path(f"{split_name}/motions")
    wav_dir = Path(f"{split_name}/wavs")

    for sequence in split_list:
        motion_path = base_path / 'motions' / f"{sequence}.pkl"
        wav_path = base_path / 'wavs' / f"{sequence}.wav"

        if not motion_path.exists() or not wav_path.exists():
            print(f"File missing: {motion_path} or {wav_path}")
            continue

        with open(motion_path, "rb") as f:
            motion_data = pickle.load(f)
        
        trans = motion_data["transl"]
        pose = motion_data["pred_thetas"].reshape(motion_data["pred_thetas"].shape[0], -1)
        scale = motion_data["pred_camera"]

        out_data = {"pos": trans, "q": pose, "scale": scale}
        with open(motion_dir / f"{sequence}.pkl", "wb") as f:
            pickle.dump(out_data, f)

        shutil.copyfile(wav_path, wav_dir / f"{sequence}.wav")

def split_data(dataset_path):
    train_list = set(file_to_list("splits/train.txt"))
    test_list = set(file_to_list("splits/test.txt"))

    dataset_path = Path(dataset_path)
    train_path = Path(__file__).parent
    prepare_directories(train_path, ["train/motions", "train/wavs", "test/motions", "test/wavs"])

    for split_list, split_name in zip([train_list, test_list], ["train", "test"]):
        copy_and_process_files(dataset_path, split_name, split_list)

