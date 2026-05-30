"""
Inference module for occlusion detection.
Handles validation, testing, and prediction generation.
"""

import torch
import pandas as pd
from tqdm import tqdm
from metrics import metric_fn
from config import OUTPUT_PRED

def validate(models, validation_loader, device):
    results_list = []
    mae_sum = 0.0
    num_samples = 0

    with torch.inference_mode():

        for X, y, gender, filename in tqdm(validation_loader, total=len(validation_loader)):
            X = X.to(device)

            y = y.float().to(device)

            # ENSEMBLE PREDICTION
            y_pred = ensemble_predict(models, X)

            y_pred = torch.clamp(y_pred, 0.0, 1.0)

            batch_mae = torch.abs(y_pred - y).mean()

            mae_sum += (batch_mae.item() * len(X))

            num_samples += len(X)

            for i in range(len(X)):
                results_list.append({
                    "filename": filename[i],
                    "pred": y_pred[i].item(),
                    "target": y[i].item(),
                    "gender": float(gender[i])
                })

    results_df = pd.DataFrame(results_list)
    results_male = results_df[results_df["gender"] == 1.0]

    results_female = results_df[results_df["gender"] == 0.0]

    metric_score = metric_fn(results_female, results_male)

    mae = mae_sum / num_samples

    return results_df, mae, metric_score


def test(models, test_loader, device):
    results_list = []
    with torch.inference_mode():

        for X, filename in tqdm(test_loader, total=len(test_loader)):
            X = X.to(device)
            y_pred = ensemble_predict(models, X)
            y_pred = torch.clamp(y_pred, 0.0, 1.0)

            for i in range(len(X)):
                results_list.append({
                    "filename": filename[i],
                    "FaceOcclusion": y_pred[i].item()
                })

    results_df = pd.DataFrame(results_list)

    results_df.to_csv(
        OUTPUT_PRED,
        sep=",",
        index=False
    )

    return results_df


def ensemble_predict(models, images):

    predictions = []

    for model in models:
        model.eval()
        pred = model(images)
        predictions.append(pred)

    final_pred = (
        0.5 * predictions[0] +
        0.3 * predictions[1] +
        0.2 * predictions[2]
    )

    return final_pred