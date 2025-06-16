from torch.utils.data import DataLoader
import torch
from src.dataset import IntroDataset
from src.model import MultimodalIntroModel
from src.trainer import train

dataset = IntroDataset("data/train", "data/labels_train.json")
loader = DataLoader(dataset, batch_size=1, shuffle=True)

model = MultimodalIntroModel()
train(model, loader)

torch.save(model.state_dict(), "best_model.pth")
print("Модель успешно сохранена как best_model.pth")
