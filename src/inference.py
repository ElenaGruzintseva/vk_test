import yaml
import torch
from src.audio import extract_mfcc
from src.video import extract_clip_embeddings
from src.handlers import merge_intervals
import numpy as np
from decord import VideoReader


with open("configs/config.yml") as f:
    config = yaml.safe_load(f)


def segment_video(video_path, clip_length_sec=5, overlap_sec=2.5):
    sr = config["sr"]
    hop_length = config["hop_length"]
    fps = config["fps"]
    frames_per_clip = config["n_frames"]
    step_frames = int((clip_length_sec - overlap_sec) * fps)

    try:
        vr = VideoReader(video_path)
        total_frames = len(vr)
    except Exception as e:
        print(f" Не могу открыть видео {video_path}: {e}")
        return []

    try:
        full_audio = extract_mfcc(video_path)
    except Exception as e:
        print(f" Ошибка при извлечении аудио для {video_path}: {e}")
        return []

    clips = []

    for start_frame in range(0, total_frames - frames_per_clip + 1, step_frames):
        end_frame = start_frame + frames_per_clip
        frame_indices = np.linspace(start_frame, end_frame, frames_per_clip).astype(int)

        start_sec = start_frame / fps
        end_sec = end_frame / fps

        audio_start = int(start_sec * sr // hop_length)
        audio_end = int(end_sec * sr // hop_length)
        audio_clip = full_audio[audio_start:audio_end]

        if len(audio_clip) < 100:
            audio_clip = np.pad(audio_clip, ((0, 100 - len(audio_clip)), (0, 0)), mode='constant')

        video_clip = extract_clip_embeddings(video_path, frame_indices)
        if video_clip is None:
            video_clip = np.zeros((8, 512))
        clips.append({
            "audio": torch.tensor(audio_clip, dtype=torch.float32),
            "video": torch.tensor(video_clip, dtype=torch.float32),
            "time": (start_sec, end_sec)
        })

    return clips


def infer(model, video_path, threshold=0.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    try:
        all_clips = segment_video(video_path)
    except Exception as e:
        print(f"Ошибка при разбиении видео {video_path}: {e}")
        return []

    intervals = []

    with torch.no_grad():
        for clip in all_clips:
            if clip["audio"] is None or clip["video"] is None:
                print("Пропуск клипа: один из сигналов пуст")
                continue

            audio = clip["audio"].unsqueeze(0).to(device)
            video = clip["video"].unsqueeze(0).to(device)
            start_sec, end_sec = clip["time"]

            logits = model(audio, video)
            prob = torch.sigmoid(logits).item()

            if prob > threshold:
                intervals.append((start_sec, end_sec))

    return merge_intervals(intervals)
