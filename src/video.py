from decord import VideoReader
from sentence_transformers import SentenceTransformer
import numpy as np
from PIL import Image


video_model = SentenceTransformer('clip-ViT-B-32')


def extract_clip_embeddings(video_path, frame_indices=None):
    vr = VideoReader(video_path)
    if frame_indices is None:
        frame_indices = np.linspace(0, len(vr) - 1, 8).astype(int)
    frames = vr.get_batch(frame_indices).asnumpy()
    images = [Image.fromarray(f) for f in frames]
    embeddings = video_model.encode(images)
    return embeddings
