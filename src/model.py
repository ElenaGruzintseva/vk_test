import torch
import torch.nn as nn


class MultimodalIntroModel(nn.Module):
    def __init__(self, audio_dim=13, video_dim=512, hidden_dim=256):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.video_proj = nn.Linear(video_dim, hidden_dim)

        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4),
            num_layers=1
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, audio_seq, video_seq):

        print("audio_seq shape:", audio_seq.shape)
        print("video_seq shape:", video_seq.shape)

        audio_emb = self.audio_proj(audio_seq)
        video_emb = self.video_proj(video_seq)

        print("audio_emb shape:", audio_emb.shape)
        print("video_emb shape:", video_emb.shape)

        audio_global = audio_emb.mean(dim=1)
        video_global = video_emb.mean(dim=1)

        print("audio_global shape:", audio_global.shape)
        print("video_global shape:", video_global.shape)
        print(type(audio_emb))
        print(isinstance(audio_emb, torch.Tensor))

        combined = torch.cat([audio_global, video_global], dim=1)
        print("Combined shape:", combined.shape)

        logits = self.classifier(combined)

        return logits
