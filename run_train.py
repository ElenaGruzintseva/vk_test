import yaml
from torch.utils.data import DataLoader
import torch
from src.dataset import IntroDataset
from src.model import MultimodalIntroModel
from src.trainer import train


with open("configs/config.yml") as f:
    config = yaml.safe_load(f)

dataset = IntroDataset(
    root_dir=config["data"]["train"],
    labels_file=config["labels_train"],
)

loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

model = MultimodalIntroModel()
train(model, loader, epochs=config["epochs"])

torch.save(model.state_dict(), "best_model.pth")
print("Модель успешно сохранена как best_model.pth")
