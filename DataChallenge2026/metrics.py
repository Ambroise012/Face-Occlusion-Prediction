"""
Metrics module for occlusion detection.
Contains evaluation metrics and scoring functions.
"""

import numpy as np
import pandas as pd

def error_fn(df):
    """
    Calculate weighted error for a DataFrame of predictions.

    Args:
        df: DataFrame with columns 'pred' and 'target'

    Returns:
        float: Weighted error score
    """
    pred = df.loc[:, "pred"]
    ground_truth = df.loc[:, "target"]
    weight = 1/30 + ground_truth

    return np.sum(((pred - ground_truth)**2) * weight, axis=0) / np.sum(weight, axis=0)

def metric_fn(female_df, male_df):
    """
    Calculate final metric combining male and female errors.

    Args:
        female_df: DataFrame with predictions for female samples
        male_df: DataFrame with predictions for male samples

    Returns:
        float: Combined metric score
    """
    err_male = error_fn(male_df)
    err_female = error_fn(female_df)

    return (err_male + err_female) / 2 + abs(err_male - err_female)

def evaluate_results(results_df):
    """
    Evaluate results DataFrame and return metric score.

    Args:
        results_df: DataFrame with columns 'pred', 'target', 'gender'

    Returns:
        float: Final metric score
    """
    results_male = results_df.loc[results_df["gender"] == 1.0]
    results_female = results_df.loc[results_df["gender"] == 0.0]

    return metric_fn(results_female, results_male)