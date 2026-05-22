"""
Training module for occlusion detection.
Handles the training loop and model optimization.
"""

import torch
from tqdm import tqdm
from config import NUM_EPOCHS

def train_model(model, training_loader, validation_loader, loss_fn, optimizer, device):
    """
    Train the model for a specified number of epochs.

    Args:
        model: PyTorch model
        training_loader: DataLoader for training data
        validation_loader: DataLoader for validation data
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device (CPU or CUDA)

    Returns:
        model: Trained model
    """
    model.train()

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for batch_idx, (X, y, gender, filename) in (pbar := tqdm(
            enumerate(training_loader),
            total=len(training_loader)
        )):
            # Transfer to GPU
            X, y = X.to(device), y.to(device)
            y = y.view(-1, 1)

            # Forward pass
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            # Check for NaN loss
            if loss.isnan():
                print(f"NaN loss encountered at batch {batch_idx}")
                print(f"Filename: {filename}")
                print(f"Label: {y}")
                print(f"Prediction: {y_pred}")
                break

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar
            pbar.set_description(f"Loss: {loss.item():.4f}")

    return model