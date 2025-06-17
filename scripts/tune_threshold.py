import numpy as np
from sklearn.metrics import f1_score
from src.inference import infer
import json
import os


def tune_threshold(model, val_dir, val_labels_file):
    with open(val_labels_file) as f:
        raw_labels = json.load(f)

    val_files = []
    true_labels = []

    for key in raw_labels:
        filename = f"{key}.mp4"
        video_path = os.path.join(val_dir, filename)
        if not os.path.exists(video_path):
            print(f"Файл {video_path} не найден, пропускаем")
            continue

        label = raw_labels[key]
        is_true_intro = 1 if label["start"] < label["end"] else 0

        val_files.append(filename)
        true_labels.append(is_true_intro)

    thresholds = np.linspace(0.1, 0.9, 9)
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in thresholds:
        preds = []
        actual_true = []

        for filename in val_files:
            video_path = os.path.join(val_dir, filename)
            label = raw_labels.get(filename.replace(".mp4", ""), {"start": 0, "end": 0})
            pred_intervals = infer(model, video_path, threshold=threshold)

            is_pred_intro = 1 if pred_intervals else 0
            is_true_intro = 1 if label["start"] < label["end"] else 0

            preds.append(is_pred_intro)
            actual_true.append(is_true_intro)

        try:
            f1 = f1_score(actual_true, preds)
            print(f"Threshold: {threshold:.2f} → F1: {f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        except Exception as e:
            print(f"Ошибка при вычислении F1 для порога {threshold}: {e}")

    return best_threshold, best_f1
