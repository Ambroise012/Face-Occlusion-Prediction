"""
Main script for occlusion detection project.
Orchestrates data loading, training, validation, and testing.
"""
from src.dataset import load_data, create_datasets, create_dataloaders
from src.model import get_model, get_loss_and_optimizer
from src.train import train_model
from src.inference import validate, test, ensemble_predict
from metrics import evaluate_results, error_fn
import torch

from config import NUM_EPOCHS, OUTPUT_PRED, PATIENCE, MODEL_NAME


def main():
    """Main execution function."""
    print("Loading and preparing data...")

    # Load data
    train_df, val_df, test_df = load_data()

    training_set, validation_set, test_set = create_datasets(
        train_df,
        val_df,
        test_df
    )

    training_loader, validation_loader, test_loader = create_dataloaders(
        training_set,
        validation_set,
        test_set,
        train_df
    )

    all_models = []
    histories = []

    for model_name in MODEL_NAME:
        print(f"\nTraining {model_name}")

        model, device = get_model(model_name)

        loss_fn, optimizer, scheduler = (
            get_loss_and_optimizer(model)
        )

        model, history = train_model(
            model=model,
            training_loader=training_loader,
            validation_loader=validation_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=NUM_EPOCHS,
            patience=PATIENCE,
            min_delta=0.0005
        )

        torch.save(model.state_dict(), f"{model_name}.pth")

        all_models.append(model)
        histories.append(history)

    # VALIDATION
    val_results, val_mae, metric_score = validate(
        all_models,
        validation_loader,
        device
    )

    # SPLIT METRICS
    male_results = val_results[val_results["gender"] == 1.0]
    female_results = val_results[val_results["gender"] == 0.0]

    male_metric = error_fn(male_results)
    female_metric = error_fn(female_results)
    fairness_gap = abs(male_metric - female_metric)

    # OCCLUSION BIN ANALYSIS
    low_occ = val_results[val_results["target"] < 0.1]

    medium_occ = val_results[
        (val_results["target"] >= 0.1)
        &
        (val_results["target"] < 0.4)
    ]

    high_occ = val_results[
        val_results["target"] >= 0.4
    ]

    low_occ_mae = (
        abs(low_occ["pred"] - low_occ["target"]).mean()
        if len(low_occ) > 0
        else 0
    )

    medium_occ_mae = (
        abs(medium_occ["pred"] - medium_occ["target"]).mean()
        if len(medium_occ) > 0
        else 0
    )

    high_occ_mae = (
        abs(high_occ["pred"] - high_occ["target"]).mean()
        if len(high_occ) > 0
        else 0
    )

    # VALIDATION RESULTS
    print("\n====================================")
    print("VALIDATION")
    print("====================================")
    print(f"\nValidation MAE: {val_mae * 100:.2f}%")
    print(f"Challenge Metric: {metric_score:.6f}")
    print(f"\nMale Metric: {male_metric:.6f}")
    print(f"Female Metric: {female_metric:.6f}")
    print(f"Fairness Gap: {fairness_gap:.6f}")

    # OCCLUSION ANALYSIS
    print("\n====================================")
    print("OCCLUSION BIN ANALYSIS")
    print("====================================")
    print(f"\nLow Occlusion (<10%) MAE: {low_occ_mae * 100:.2f}%")
    print(f"Medium Occlusion (10-40%) MAE: {medium_occ_mae * 100:.2f}%")
    print(f"High Occlusion (>40%) MAE: {high_occ_mae * 100:.2f}%")
    print(f"\nLow Occlusion Samples: {len(low_occ)}")
    print(f"Medium Occlusion Samples: {len(medium_occ)}")
    print(f"High Occlusion Samples: {len(high_occ)}")

    # PREDICTION DISTRIBUTION
    print("\n====================================")
    print("PREDICTION DISTRIBUTION")
    print("====================================")
    print(f"\nTarget Mean: {val_results['target'].mean():.4f}")
    print(f"Prediction Mean: {val_results['pred'].mean():.4f}")
    print(f"\nTarget Max: {val_results['target'].max():.4f}")
    print(f"Prediction Max: {val_results['pred'].max():.4f}")

    # TEST INFERENCE
    print("\n====================================")
    print("TEST INFERENCE")
    print("====================================")
    test(all_models, test_loader, device)

    print(f"\nPredictions saved to:")
    print(OUTPUT_PRED)

    # TRAINING SUMMARY
    print("\n====================================")
    print("TRAINING SUMMARY")
    print("====================================")

    for idx, model_name in enumerate(MODEL_NAME):

        print(f"\nModel: {model_name}")

        model_history = histories[idx]

        best_train_mae = min(
            model_history["train_mae"]
        )

        best_val_mae = min(
            model_history["val_mae"]
        )

        generalization_gap = (
            best_val_mae - best_train_mae
        ) * 100

        print(
            f"Best Train MAE: "
            f"{best_train_mae * 100:.2f}%"
        )

        print(
            f"Best Validation MAE: "
            f"{best_val_mae * 100:.2f}%"
        )

        print(
            f"Generalization Gap: "
            f"{generalization_gap:.2f}%"
        )

        if generalization_gap > 5:
            print(
                "WARNING: possible overfitting."
            )

    print("\n========== DONE ==========")

if __name__ == "__main__":
    main()