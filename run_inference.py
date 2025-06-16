import os
import argparse
from src.inference import infer
import torch
import json
from src.model import MultimodalIntroModel


def main():
    parser = argparse.ArgumentParser(description="Detect intro in TV series")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold")
    args = parser.parse_args()

    model = MultimodalIntroModel()

    data = torch.load("best_model.pth", weights_only=False)
    print(type(data))

    model.load_state_dict(torch.load("best_model.pth", weights_only=False))
    model.eval()

    test_dir = "data/test"
    result = {}

    for filename in os.listdir(test_dir):
        if filename.endswith(".mp4"):
            video_path = os.path.join(test_dir, filename)
            print(f"Processing {filename}...")

            intervals = infer(model, video_path, threshold=args.threshold)
            result[filename] = {
                "intervals": [[int(s), int(e)] for s, e in intervals]
            }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
