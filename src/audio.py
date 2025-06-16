import librosa
import ffmpeg


def extract_mfcc(audio_path, sr=22050, n_mfcc=13, hop_length=512):
    if audio_path.endswith(".mp4"):
        output_path = audio_path + ".wav"
        (
            ffmpeg
            .input(audio_path)
            .output(output_path, format='wav', acodec='pcm_s16le', ac=1, ar=sr)
            .run(overwrite_output=True, capture_stderr=True)
        )
        audio_path = output_path

    y, sr = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    return mfcc.T
