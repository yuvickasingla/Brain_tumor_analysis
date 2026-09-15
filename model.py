import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import config



class BrainTumorCNN(nn.Module):
    """4-block custom CNN trained from scratch."""

    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(3,  32,  3, padding=1); self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64,  3, padding=1); self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1); self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,256, 3, padding=1); self.bn4 = nn.BatchNorm2d(256)
        self.pool  = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        flatten = 256 * 14 * 14          
        self.fc1 = nn.Linear(flatten, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def _block(self, x, conv, bn):
        return self.pool(F.relu(bn(conv(x))))

    def forward(self, x):
        x = self._block(x, self.conv1, self.bn1)
        x = self._block(x, self.conv2, self.bn2)
        x = self._block(x, self.conv3, self.bn3)
        x = self._block(x, self.conv4, self.bn4)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

    def get_last_conv_layer(self):
        return self.conv4

    def freeze(self):   pass
    def unfreeze(self): pass



class ResNet50Model(nn.Module):

    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        backbone = models.resnet50(weights=weights)

        # Replace the final FC layer
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        self.backbone = backbone
        self.freeze()   # start in Phase 1

    def forward(self, x):
        return self.backbone(x)

    def freeze(self):
        for name, param in self.backbone.named_parameters():
            param.requires_grad = ('fc' in name)

    def unfreeze(self):
        for name, param in self.backbone.named_parameters():
            if any(k in name for k in ['layer3', 'layer4', 'fc']):
                param.requires_grad = True

    def get_last_conv_layer(self):
        return self.backbone.layer4[-1].conv3



class EfficientNetModel(nn.Module):

    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        weights  = models.EfficientNet_B3_Weights.IMAGENET1K_V1
        backbone = models.efficientnet_b3(weights=weights)

        # Replace classifier
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        self.backbone = backbone
        self.freeze()

    def forward(self, x):
        return self.backbone(x)

    def freeze(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for i, block in enumerate(self.backbone.features):
            if i >= 5:
                for param in block.parameters():
                    param.requires_grad = True

    def get_last_conv_layer(self):
        # Last conv in the features sequence
        return self.backbone.features[-1][0]


class MobileNetV3Model(nn.Module):

    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        weights  = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        backbone = models.mobilenet_v3_large(weights=weights)

        # Replace classifier (last Linear layer)
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        self.backbone = backbone
        self.freeze()   # start in Phase 1

    def forward(self, x):
        return self.backbone(x)

    def freeze(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def unfreeze(self):
        num_blocks = len(self.backbone.features)
        for i, block in enumerate(self.backbone.features):
            if i >= num_blocks - 3:
                for param in block.parameters():
                    param.requires_grad = True

    def get_last_conv_layer(self):
        # Last conv in the features sequence
        return self.backbone.features[-1][0]

MODEL_REGISTRY = {
    'custom_cnn':   BrainTumorCNN,
    'resnet50':     ResNet50Model,
    'efficientnet': EfficientNetModel,
    'mobilenetv3':   MobileNetV3Model,
}

def get_model(name: str, num_classes: int = config.NUM_CLASSES) -> nn.Module:
    
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](num_classes=num_classes)


def count_params(model: nn.Module):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params    : {total:,}")
    print(f"  Trainable params: {trainable:,}")


if __name__ == '__main__':
    dummy = torch.randn(2, 3, 224, 224)
    for name in MODEL_REGISTRY:
        m = get_model(name)
        out = m(dummy)
        print(f"\n{name}  → output shape: {out.shape}")
        count_params(m)
        m.unfreeze()
        print(f"  After unfreeze:")
        count_params(m)
    print("\nAll model checks passed ✓")
