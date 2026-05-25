"""
Model module for occlusion detection.
Defines the neural network architecture.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from config import MODEL_NAME, USE_CUDA


class OcclusionRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        if MODEL_NAME == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT
            self.backbone = models.efficientnet_b0(
                weights=weights
            )

            in_features = (
                self.backbone.classifier[1].in_features
            )

            self.backbone.classifier[1] = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

        else:
            raise ValueError(
                f"Unknown model: {MODEL_NAME}"
            )

    def forward(self, x):

        return self.backbone(x).squeeze(1)


def get_model():
    model = OcclusionRegressor()

    if USE_CUDA and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    model = model.to(device)

    return model, device


class WeightedSmoothL1Loss(nn.Module):
    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = beta
        self.loss_fn = nn.SmoothL1Loss(
            reduction='none',
            beta=beta
        )

    def forward(self, pred, target):
        base_loss = self.loss_fn(pred, target)
        # stronger weight for high occlusion
        weights = 1.0 + (4.0 * target)
        loss = weights * base_loss
        return loss.mean()


def get_loss_and_optimizer(model):
    loss_fn = WeightedSmoothL1Loss(beta=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=10
    )

    return loss_fn, optimizer, scheduler