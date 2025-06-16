import torch
import torch.optim as optim
from tqdm import tqdm


def train(model, dataloader, epochs=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader):
            audio = batch["audio"].to(device)
            video = batch["video"].to(device)
            label = batch["label"].to(device)

            logits = model(audio, video)
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
