import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from src.audio import extract_mfcc
from src.video import extract_clip_embeddings


class IntroDataset(Dataset):
    def __init__(self, root_dir, labels_file, sr=22050):
        self.root_dir = root_dir
        self.sr = sr

        with open(labels_file) as f:
            raw_labels = json.load(f)

        self.labels = {}
        for key in raw_labels:
            filename = key + ".mp4"
            start = self.time_to_seconds(raw_labels[key]["start"])
            end = self.time_to_seconds(raw_labels[key]["end"])
            self.labels[filename] = {"start": start, "end": end}

        self.files = [f for f in os.listdir(root_dir) if f.endswith(".mp4")]

    @staticmethod
    def time_to_seconds(t: str) -> int:
        parts = list(map(int, t.split(":")))
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2:
            return parts[0] * 60 + parts[1]
        else:
            return int(parts[0])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        path = os.path.join(self.root_dir, filename)
        label = self.labels.get(filename, {"start": 0, "end": 0})

        try:
            audio_features = extract_mfcc(path, sr=self.sr)
        except Exception as e:
            print(f"Ошибка при обработке аудио {filename}: {e}")
            audio_features = np.zeros((100, 13))

        try:
            video_features = extract_clip_embeddings(path)
        except Exception as e:
            print(f"Ошибка при обработке видео {filename}: {e}")
            video_features = torch.zeros((8, 512))

        is_intro = 1 if label["start"] <= 0 < label["end"] else 0

        return {
            "audio": torch.tensor(audio_features, dtype=torch.float32),
            "video": torch.tensor(video_features, dtype=torch.float32),
            "label": torch.tensor([is_intro], dtype=torch.float32)
        }
