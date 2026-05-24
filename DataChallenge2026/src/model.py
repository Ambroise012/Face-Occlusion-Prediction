"""
Model module for occlusion detection.
Defines the neural network architecture.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from config import MODEL_NAME, NUM_CLASSES, USE_CUDA

def get_model():
    """
    Initialize and configure the model.

    Returns:
        model: PyTorch model ready for training
        device: Device (CPU or CUDA) the model is on
    """
    # Initialize model
    if MODEL_NAME == "efficientnet_b0":

        # pretrained weights
        weights = models.EfficientNet_B0_Weights.DEFAULT

        model = models.efficientnet_b0(weights=weights)

        in_features = model.classifier[1].in_features

        # Regression head
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    else:
        raise ValueError(f"Unknown model: {MODEL_NAME}")

    # Configure CUDA
    if USE_CUDA and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    model = model.to(device)
    return model, device

def get_loss_and_optimizer(model):
    """
    Get loss function and optimizer for training.

    Args:
        model: PyTorch model

    Returns:
        loss_fn: Loss function
        optimizer: Optimizer
    """
    # loss_fn = nn.MSELoss()
    loss_fn = nn.SmoothL1Loss(beta=0.1)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=10
    )
    
    return loss_fn, optimizer