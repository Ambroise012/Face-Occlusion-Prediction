"""
Inference module for occlusion detection.
Handles validation, testing, and prediction generation.
"""

import torch
import pandas as pd
from tqdm import tqdm
from metrics import error_fn, metric_fn
from config import OUTPUT_PRED

def validate(model, validation_loader, device):
    """
    Run validation and return results DataFrame with predictions.

    Args:
        model: PyTorch model
        validation_loader: DataLoader for validation data
        device: Device (CPU or CUDA)

    Returns:
        results_df: DataFrame with validation results
        metric_score: Final metric score
    """
    model.eval()
    results_list = []

    with torch.inference_mode():
        for batch_idx, (X, y, gender, filename) in tqdm(
            enumerate(validation_loader),
            total=len(validation_loader)
        ):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            for i in range(len(X)):
                results_list.append({
                    'filename': filename[i],
                    'pred': float(y_pred[i]),
                    'target': float(y[i]),
                    'gender': float(gender[i])
                })

    results_df = pd.DataFrame(results_list)

    # Calculate metric
    results_male = results_df.loc[results_df["gender"] == 1.0]
    results_female = results_df.loc[results_df["gender"] == 0.0]
    metric_score = metric_fn(results_female, results_male)

    return results_df, metric_score

def test(model, test_loader, device):
    """
    Run inference on test set and save predictions.

    Args:
        model: PyTorch model
        test_loader: DataLoader for test data
        device: Device (CPU or CUDA)

    Returns:
        results_df: DataFrame with test predictions
    """
    model.eval()
    results_list = []

    with torch.inference_mode():
        for batch_idx, (X, filename) in tqdm(
            enumerate(test_loader),
            total=len(test_loader)
        ):
            X = X.to(device)
            y_pred = model(X)

            for i in range(len(X)):
                results_list.append({
                    'filename': filename[i],
                    'FaceOcclusion': float(y_pred[i]),
                })

    results_df = pd.DataFrame(results_list)
    results_df['gender'] = 'x'  # Placeholder for test set

    # Save predictions
    results_df.to_csv(OUTPUT_PRED, sep=',', index=False)

    return results_df