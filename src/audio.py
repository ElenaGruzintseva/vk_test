import yaml
import librosa
import ffmpeg
import numpy as np


with open("configs/config.yml") as f:
    config = yaml.safe_load(f)


def extract_mfcc(audio_path):
    sr = config["sr"]
    n_mfcc = config["n_mfcc"]
    hop_length = config["hop_length"]

    if audio_path.endswith(".mp4"):
        output_path = audio_path + ".wav"
        (
            ffmpeg
            .input(audio_path)
            .output(output_path, format='wav', acodec='pcm_s16le', ac=1, ar=sr)
            .run(overwrite_output=True, capture_stderr=True)
        )
        audio_path = output_path

    try:
        y, sr = librosa.load(audio_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        return mfcc.T
    except Exception as e:
        print(f"Ошибка при извлечении аудио признаков: {e}")
        return np.zeros((100, 13))
