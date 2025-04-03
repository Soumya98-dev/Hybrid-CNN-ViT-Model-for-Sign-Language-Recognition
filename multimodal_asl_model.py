import torch
import torch.nn as nn
from torchvision.models import vit_b_16

class LandmarkMLP(nn.Module):
    def __init__(self, input_dim=63, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class ViTEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.vit = vit_b_16(pretrained=pretrained)
        self.vit.heads = nn.Identity()

    def forward(self, x):
        return self.vit(x)

class AttentionFusion(nn.Module):
    def __init__(self, visual_dim=768, landmark_dim=256, fusion_dim=512, num_classes=100):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, fusion_dim)
        self.landmark_proj = nn.Linear(landmark_dim, fusion_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=fusion_dim, nhead=8, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.cls_token = nn.Parameter(torch.randn(1, 1, fusion_dim))
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, visual_feat, landmark_feat):
        B = visual_feat.size(0)
        visual_proj = self.visual_proj(visual_feat).unsqueeze(1)
        landmark_proj = self.landmark_proj(landmark_feat).unsqueeze(1)
        cls = self.cls_token.expand(B, -1, -1)
        sequence = torch.cat([cls, visual_proj, landmark_proj], dim=1)
        encoded = self.transformer_encoder(sequence)
        return self.classifier(encoded[:, 0])

class MultiModalAttentionModel(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.visual_encoder = ViTEncoder()
        self.landmark_encoder = LandmarkMLP()
        self.fusion = AttentionFusion(num_classes=num_classes)

    def forward(self, image, landmarks):
        visual_feat = self.visual_encoder(image)
        landmark_feat = self.landmark_encoder(landmarks)
        return self.fusion(visual_feat, landmark_feat)
