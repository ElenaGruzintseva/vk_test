import numpy as np
import torch
from src.audio import extract_mfcc
from src.video import extract_clip_embeddings
from src.utils import merge_intervals
from decord import VideoReader


def segment_video(video_path, audio_window=100, video_window=8, overlap=0.5):
    full_audio = extract_mfcc(video_path)

    vr = VideoReader(video_path)
    total_frames = len(vr)
    frame_indices_list = []

    step = int((1 - overlap) * audio_window)
    clip_times = []

    clips = []
    for i in range(0, len(full_audio) - audio_window + 1, step):
        start_sec = i * 0.5
        end_sec = start_sec + 5
        clip_times.append((start_sec, end_sec))

        audio_clip = full_audio[i:i + audio_window]
        frame_indices = np.linspace(
            int(start_sec * vr.get_avg_fps()),
            int(end_sec * vr.get_avg_fps()),
            video_window
        ).astype(int)
        video_clip = extract_clip_embeddings(video_path, frame_indices)

        yield {
            "audio": torch.tensor(audio_clip, dtype=torch.float32).unsqueeze(0),
            "video": torch.tensor(video_clip, dtype=torch.float32).unsqueeze(0),
            "time": (start_sec, end_sec)
        }


def infer(model, video_path, threshold=0.5, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device)
    model.eval()

    audio_features = extract_mfcc(video_path)
    vr = VideoReader(video_path)
    fps = vr.get_avg_fps()

    frame_indices_list = []
    step = 5
    clip_length_sec = 5
    total_frames = len(vr)
    frames_per_clip = int(clip_length_sec * fps)
    step_frames = int(step * fps)

    all_intervals = []

    for start_frame in range(0, total_frames - frames_per_clip + 1, step_frames):
        end_frame = start_frame + frames_per_clip
        start_sec = start_frame / fps
        end_sec = end_frame / fps

        frame_indices = np.linspace(start_frame, end_frame, 8).astype(int)
        video_clip = extract_clip_embeddings(video_path, frame_indices)

        audio_clip = audio_features[int(start_frame / fps * 2): int(end_frame / fps * 2)]
        if len(audio_clip) < 100:
            audio_clip = np.pad(audio_clip, ((0, 100 - len(audio_clip)), (0, 0)), mode='constant')

        with torch.no_grad():
            logits = model(
                torch.tensor(audio_clip, dtype=torch.float32).unsqueeze(0).to(device),
                torch.tensor(video_clip, dtype=torch.float32).unsqueeze(0).to(device)
            )
            prob = torch.sigmoid(logits).item()

        if prob > threshold:
            all_intervals.append((start_sec, end_sec))

    return merge_intervals(all_intervals)
