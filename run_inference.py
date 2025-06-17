import os
import argparse
from src.inference import infer
import torch
from src.model import MultimodalIntroModel
import json


def main():
    parser = argparse.ArgumentParser(description="Detect intro in TV series")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold")
    parser.add_argument("--save_to", type=str, default="results.json", help="Output file")
    args = parser.parse_args()

    model = MultimodalIntroModel()
    model.load_state_dict(torch.load("best_model.pth", weights_only=False))
    model.eval()
    print(" Модель загружена")

    test_dir = "data/test"
    result = {}

    print(f"🔍 Обрабатываем файлы из {test_dir}")
    for filename in os.listdir(test_dir):
        if filename.endswith(".mp4"):
            video_path = os.path.join(test_dir, filename)
            print(f" Обработка: {filename}...")

            try:
                intervals = infer(model, video_path, threshold=args.threshold)
                print(f" Найденные интервалы: {intervals}")
            except Exception as e:
                print(f"Ошибка при инференсе {filename}: {e}")
                if intervals:
                    result[filename] = {
                        "intervals": [[int(s), int(e)] for s, e in intervals]
                    }
                    print(f" Заставка найдена: {filename} → {intervals}")
                else:
                    result[filename] = {"intervals": []}
                    print(f" Заставка НЕ найдена: {filename}")
                result[filename] = {
                    "intervals": [[int(s), int(e)] for s, e in intervals] if intervals else []
                }

    with open(args.save_to, "w") as f:
        json.dump(result, f, indent=2)

    print(f" Результаты сохранены в {args.save_to}")


if __name__ == "__main__":
    main()
