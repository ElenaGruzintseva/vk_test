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

            if label is None or torch.isnan(label).any():
                print("⚠️ Пропуск батча — метка некорректна")
                continue

            logits = model(audio, video)

            prob = torch.sigmoid(logits).item()

            print(f"Logits: {logits.item()}, Prob: {prob:.4f}, Label: {label.item()}")

            if logits is None or torch.isnan(logits).any():
                print("⚠️ Пропуск батча — логиты некорректны")
                continue

            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
