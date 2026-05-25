"""
Inference module for occlusion detection.
Handles validation, testing, and prediction generation.
"""

import torch
import pandas as pd
from tqdm import tqdm
from metrics import error_fn, metric_fn
from config import OUTPUT_PRED

# def validate(model, validation_loader, device):
#     """
#     Run validation and return results DataFrame with predictions.

#     Args:
#         model: PyTorch model
#         validation_loader: DataLoader for validation data
#         device: Device (CPU or CUDA)

#     Returns:
#         results_df: DataFrame with validation results
#         metric_score: Final metric score
#     """
#     model.eval()
#     results_list = []

#     with torch.inference_mode():
#         for batch_idx, (X, y, gender, filename) in tqdm(
#             enumerate(validation_loader),
#             total=len(validation_loader)
#         ):
#             X, y = X.to(device), y.to(device)
#             y_pred = model(X)

#             for i in range(len(X)):
#                 results_list.append({
#                     'filename': filename[i],
#                     'pred': float(y_pred[i]),
#                     'target': float(y[i]),
#                     'gender': float(gender[i])
#                 })

#     results_df = pd.DataFrame(results_list)

#     # Calculate metric
#     results_male = results_df.loc[results_df["gender"] == 1.0]
#     results_female = results_df.loc[results_df["gender"] == 0.0]
#     metric_score = metric_fn(results_female, results_male)

#     return results_df, metric_score

def validate(model, validation_loader, device):
    model.eval()
    results_list = []
    mae_sum = 0.0
    num_samples = 0

    with torch.inference_mode():
        for X, y, gender, filename in tqdm(
            validation_loader,
            total=len(validation_loader)
        ):
            X = X.to(device)
            y = (
                y.float()
                .unsqueeze(1)
                .to(device)
            )
            # PREDICTIONS
            y_pred = model(X)

            y_pred = torch.clamp(
                y_pred,
                0.0,
                1.0
            )

            # MAE
            batch_mae = torch.abs(y_pred - y).mean()

            mae_sum += batch_mae.item() * len(X)

            num_samples += len(X)

            # STORE RESULTS
            for i in range(len(X)):
                results_list.append({
                    "filename": filename[i],
                    "pred": y_pred[i].item(),
                    "target": y[i].item(),
                    "gender": float(gender[i])
                })

    # DATAFRAME
    results_df = pd.DataFrame(results_list)

    # CHALLENGE METRIC
    results_male = results_df[
        results_df["gender"] == 1.0
    ]

    results_female = results_df[
        results_df["gender"] == 0.0
    ]

    metric_score = metric_fn(
        results_female,
        results_male
    )

    mae = mae_sum / num_samples

    return results_df, mae, metric_score

# def test(model, test_loader, device):
#     model.eval()
#     results_list = []

#     with torch.inference_mode():
#         for batch_idx, (X, filename) in tqdm(
#             enumerate(test_loader),
#             total=len(test_loader)
#         ):
#             X = X.to(device)
#             y_pred = model(X)

#             for i in range(len(X)):
#                 results_list.append({
#                     'filename': filename[i],
#                     'FaceOcclusion': float(y_pred[i]),
#                 })

#     results_df = pd.DataFrame(results_list)

#     # Save predictions
#     results_df.to_csv(OUTPUT_PRED, sep=',', index=False)

#     return results_df


def test(model, test_loader, device):
    model.eval()
    results_list = []

    with torch.inference_mode():
        for X, filename in tqdm(
            test_loader,
            total=len(test_loader)
        ):
            X = X.to(device)
            y_pred = model(X)
            y_pred = torch.clamp(
                y_pred,
                0.0,
                1.0
            )
            for i in range(len(X)):
                results_list.append({
                    "filename": filename[i],
                    "FaceOcclusion": y_pred[i].item()
                })

    results_df = pd.DataFrame(
        results_list
    )
    results_df.to_csv(
        OUTPUT_PRED,
        sep=",",
        index=False
    )

    return results_df
