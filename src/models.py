# src/models.py
# All ACDNet model components in one file so the notebook can import cleanly.
# Contents: AnatomyCNN, CBAM, FiLM, SegmentationHead, ACDNet, build_acdnet

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from efficientnet_pytorch import EfficientNet
    _EFFNET_SRC = "efficientnet_pytorch"
except ImportError:
    from torchvision.models import efficientnet_b0 as _effnet_b0
    _EFFNET_SRC = "torchvision"

BACKBONE_CHANNELS = 1280   # EfficientNet-B0 final feature channels


# ─── Stage 2: Anatomy branch CNN ──────────────────────────────────────────────

class AnatomyCNN(nn.Module):
    """3-layer CNN → 64-dim location embedding + 3-class logits."""
    def __init__(self, num_classes: int = 3, embedding_dim: int = 64, dropout_p: float = 0.4):
        # 🔥 FIX #4: Add configurable dropout to prevent overfitting
        super().__init__()
        self.embedding_dim = embedding_dim
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))          # 224→112
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))          # 112→56
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((4, 4))) # →[B,128,4,4]
        self.embedding_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),  # 🔥 CHANGED: from hardcoded 0.3 to configurable dropout_p
            nn.Linear(256, embedding_dim), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        emb    = self.embedding_layer(x)
        logits = self.classifier(emb)
        return logits, emb

    def get_embedding(self, x):
        _, emb = self.forward(x)
        return emb


def build_anatomy_cnn(num_classes=3, embedding_dim=64, dropout_p=0.4, checkpoint_path=None):
    # 🔥 FIX #4: Accept dropout_p parameter for regularization
    model = AnatomyCNN(num_classes, embedding_dim, dropout_p=dropout_p)
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        print(f"[INFO] Anatomy CNN loaded from {checkpoint_path}")
    return model


def freeze_anatomy_cnn(model):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    print("[INFO] Anatomy CNN frozen.")


# ─── Stage 4a: CBAM ───────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        mid = max(in_channels // reduction_ratio, 1)
        self.shared_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, mid, bias=False), nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False))

    def forward(self, x):
        B, C, H, W = x.shape
        avg = F.adaptive_avg_pool2d(x, 1).view(B, C)
        mx  = F.adaptive_max_pool2d(x, 1).view(B, C)
        attn = torch.sigmoid(self.shared_mlp(avg) + self.shared_mlp(mx))
        return x * attn.view(B, C, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size-1)//2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True)[0]
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel=7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x):
        return self.spatial_att(self.channel_att(x))


# ─── Stage 4b: FiLM ───────────────────────────────────────────────────────────

class FiLM(nn.Module):
    """output = γ(embedding) × features + β(embedding)"""
    def __init__(self, feature_channels, embedding_dim=64):
        super().__init__()
        self.feature_channels = feature_channels
        self.gamma_net = nn.Linear(embedding_dim, feature_channels)
        self.beta_net  = nn.Linear(embedding_dim, feature_channels)
        nn.init.ones_(self.gamma_net.weight);  nn.init.zeros_(self.gamma_net.bias)
        nn.init.zeros_(self.beta_net.weight);  nn.init.zeros_(self.beta_net.bias)

    def forward(self, features, embedding):
        gamma = self.gamma_net(embedding).view(-1, self.feature_channels, 1, 1)
        beta  = self.beta_net(embedding).view(-1,  self.feature_channels, 1, 1)
        return gamma * features + beta


# ─── Stage 5b: Segmentation head ──────────────────────────────────────────────

class SegmentationHead(nn.Module):
    def __init__(self, in_channels=1280, output_size=224):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1))
        self.bbox_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(in_channels, 256), nn.ReLU(inplace=True),
            nn.Dropout(p=0.3), nn.Linear(256, 4), nn.Sigmoid())

    def forward(self, x):
        return self.decoder(x), self.bbox_head(x)


# ─── Full ACDNet ──────────────────────────────────────────────────────────────

class ACDNet(nn.Module):
    def __init__(self, anatomy_cnn, num_uc_grades=3,
                 embedding_dim=64, dropout_p=0.3, pretrained=True):
        super().__init__()
        self.anatomy_cnn = anatomy_cnn
        for p in self.anatomy_cnn.parameters():
            p.requires_grad = False

        if _EFFNET_SRC == "efficientnet_pytorch":
            self.backbone = (EfficientNet.from_pretrained("efficientnet-b0")
                             if pretrained else EfficientNet.from_name("efficientnet-b0"))
            self._btype = "efficientnet_pytorch"
        else:
            _m = _effnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
            self.backbone = _m.features
            self._btype = "torchvision"

        self.cbam             = CBAM(BACKBONE_CHANNELS, 16, 7)
        self.film             = FiLM(BACKBONE_CHANNELS, embedding_dim)
        self.detection_head   = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout_p), nn.Linear(BACKBONE_CHANNELS, 256),
            nn.ReLU(inplace=True), nn.Dropout(dropout_p), nn.Linear(256, 1))
        self.segmentation_head = SegmentationHead(BACKBONE_CHANNELS, 224)
        self.severity_head    = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout_p), nn.Linear(BACKBONE_CHANNELS, 256),
            nn.ReLU(inplace=True), nn.Dropout(dropout_p),
            nn.Linear(256, num_uc_grades))

    def _features(self, x):
        if self._btype == "efficientnet_pytorch":
            return self.backbone.extract_features(x)
        return self.backbone(x)

    def forward(self, x):
        features = self._features(x)
        features = self.cbam(features)
        with torch.no_grad():
            _, embedding = self.anatomy_cnn(x)
        features = self.film(features, embedding)
        det_logit          = self.detection_head(features)
        mask_logit, bbox   = self.segmentation_head(features)
        sev_logit          = self.severity_head(features)
        return {"detection_logit": det_logit, "mask_logit": mask_logit,
                "bbox": bbox, "severity_logit": sev_logit, "features": features}


def build_acdnet(anatomy_checkpoint, num_uc_grades=3,
                 embedding_dim=64, dropout_p=0.3, pretrained_backbone=True):
    anatomy_cnn = build_anatomy_cnn(
        num_classes=3,
        embedding_dim=embedding_dim,
        checkpoint_path=anatomy_checkpoint,
    )
    freeze_anatomy_cnn(anatomy_cnn)
    return ACDNet(anatomy_cnn, num_uc_grades, embedding_dim,
                  dropout_p, pretrained_backbone)
