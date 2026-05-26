"""
Model module for occlusion detection.
Defines the neural network architecture.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from config import MODEL_NAME, USE_CUDA, LR


class OcclusionRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        if MODEL_NAME == "efficientnet_b2":
            weights = models.EfficientNet_B2_Weights.DEFAULT
            self.backbone = models.efficientnet_b2(
                weights=weights
            )

            in_features = (
                self.backbone.classifier[1].in_features
            )

            self.backbone.classifier[1] = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(in_features, 512),
                nn.SiLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 1),
                nn.Sigmoid()
            )

        elif MODEL_NAME == "efficientnet_b0":
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


# class HybridOcclusionLoss(nn.Module):
#     """
#     Loss function:
#     - weighted MSE aligned with challenge metric
#     - SmoothL1 stabilizes training
#     - Strong occlusions weighted more
#     """
#     def __init__(self):
#         super().__init__()
#         self.smooth_l1 = nn.SmoothL1Loss(
#             reduction='none',
#             beta=0.05
#         )

#     def forward(
#         self,
#         pred,
#         target
#     ):
#         weights = 1/30 + 4.0 * target

#         # WEIGHTED MSE
#         mse = (pred - target) ** 2

#         # SMOOTH L1
#         l1 = self.smooth_l1(pred, target)

#         # HYBRID
#         loss = (
#             0.7 * mse
#             +
#             0.3 * l1
#         )

#         loss = weights * loss

#         return loss.mean()


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
    loss_fn = WeightedSmoothL1Loss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    # previous : CosineAnnealingLR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=10
    )
    return loss_fn, optimizer, scheduler
