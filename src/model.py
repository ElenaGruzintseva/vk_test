import yaml
import torch
import torch.nn as nn


with open("configs/config.yml") as f:
    config = yaml.safe_load(f)


class MultimodalIntroModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_proj = nn.Linear(config["audio_dim"], config["hidden_dim"])
        self.video_proj = nn.Linear(config["video_dim"], config["hidden_dim"])

        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config["hidden_dim"], nhead=4),
            num_layers=1
        )

        self.classifier = nn.Sequential(
            nn.Linear(config["hidden_dim"] * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, audio_seq, video_seq):
        audio_emb = self.audio_proj(audio_seq)
        video_emb = self.video_proj(video_seq)
        audio_global = audio_emb.mean(dim=1)
        video_global = video_emb.mean(dim=1)
        combined = torch.cat([audio_global, video_global], dim=1)

        print("Combined shape:", combined.shape)
        logits = self.classifier(combined)
        print("Logits:", logits)
        print("Logits.isnan():", torch.isnan(logits).any())
        print("Logits.isinf():", torch.isinf(logits).any())
        return logits
