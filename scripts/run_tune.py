import json
from src.model import MultimodalIntroModel
import torch
from .tune_threshold import tune_threshold


def main():
    model_path = "best_model.pth"
    val_dir = "data/val"
    val_labels_file = "data/labels_val.json"

    model = MultimodalIntroModel()
    model.load_state_dict(torch.load(model_path, weights_only=False))
    model.eval()

    best_threshold, best_f1 = tune_threshold(model, val_dir, val_labels_file)

    print(f" Лучший порог: {best_threshold:.2f}, F1-score: {best_f1:.4f}")
    with open("best_threshold.json", "w") as f:
        json.dump({"threshold": float(best_threshold), "f1": float(best_f1)}, f, indent=2)
    print("Лучший порог сохранён в best_threshold.json")


if __name__ == "__main__":
    main()
