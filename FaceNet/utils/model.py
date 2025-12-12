import torch.nn as nn
from torchvision.models import resnet50
import torch.nn.functional as F

class FaceEncoder(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        backbone = resnet50(weights="IMAGENET1K_V1")
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.embedding = nn.Linear(2048, embedding_size)

    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        return F.normalize(x, p=2, dim=1)
