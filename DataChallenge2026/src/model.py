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
    if MODEL_NAME == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {MODEL_NAME}")

    # Configure CUDA
    if USE_CUDA and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    # Move model to device
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
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    return loss_fn, optimizer