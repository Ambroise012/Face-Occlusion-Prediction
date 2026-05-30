"""
Model module for occlusion detection.
Defines the neural network architecture.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from config import MODEL_NAME, USE_CUDA, LR


class OcclusionRegressor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        if model_name == "efficientnet_b2":
            weights = models.EfficientNet_B2_Weights.DEFAULT
            self.backbone = models.efficientnet_b2(
                weights=weights
            )

            in_features = (
                self.backbone.classifier[1].in_features
            )

            self.backbone.classifier[1] = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256), # stability
                nn.SiLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1),
                # nn.Sigmoid()
            )

        elif model_name == "efficientnet_b0":
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
                nn.BatchNorm1d(128),
                nn.SiLU(),
                # nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
                # nn.Sigmoid()
            )
        elif model_name == "efficientnet_v2_s":
            weights = (models.EfficientNet_V2_S_Weights.DEFAULT)
            self.backbone = (
                models.efficientnet_v2_s(
                    weights=weights
                )
            )
            in_features = (
                self.backbone.classifier[1].in_features
            )
            self.backbone.classifier[1] = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
                # nn.Sigmoid()
            )

        elif model_name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT
            self.backbone = models.convnext_tiny(
                weights=weights
            )
            in_features = (
                self.backbone.classifier[2].in_features
            )
            self.backbone.classifier[2] = nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Linear(in_features, 256),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
                # nn.GELU(),
                # nn.Dropout(0.2),
                # nn.Linear(128, 1),
                # nn.Sigmoid()
            )
        
        else:
            raise ValueError(
                f"Unknown model: {model_name}"
            )

    def forward(self, x):
        return self.backbone(x).squeeze(1)


def get_model(model_name):
    model = OcclusionRegressor(model_name)

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

class HybridLoss(nn.Module):
    def __init__(self, beta=0.1, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta)
        self.mae = nn.L1Loss()

    def forward(self, pred, target):
        smooth = self.smooth_l1(pred, target)
        mae = self.mae(pred, target)
        return self.alpha * smooth + (1 - self.alpha) * mae

# def freeze_backbone(model):
#     """
#     Freeze pretrained backbone.
#     Useful for first few epochs.
#     """
#     for param in model.backbone.features.parameters():
#         param.requires_grad = False


# def unfreeze_backbone(model):
#     """
#     Unfreeze backbone for fine-tuning.
#     """
#     for param in model.backbone.features.parameters():
#         param.requires_grad = True



def get_loss_and_optimizer(model):
    # loss_fn = WeightedSmoothL1Loss()
    loss_fn = HybridLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    # previous : CosineAnne alingLR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=10
    )

    return loss_fn, optimizer, scheduler
