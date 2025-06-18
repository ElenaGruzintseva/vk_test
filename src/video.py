import yaml
from decord import VideoReader
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image


with open("configs/config.yml") as f:
    config = yaml.safe_load(f)

video_model = SentenceTransformer(config["clip_model_name"])


def extract_clip_embeddings(video_path, frame_indices=None):
    try:
        vr = VideoReader(video_path)
        if frame_indices is None:
            frame_indices = np.linspace(
                0, len(vr) - 1, config["n_frames"]
            ).astype(int)
        frames = vr.get_batch(frame_indices).asnumpy()
        images = [Image.fromarray(f) for f in frames]
        embeddings = video_model.encode(images)
        return embeddings
    except Exception as e:
        print(f" Ошибка при извлечении видео признаков: {e}")
        return None
