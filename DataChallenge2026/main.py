"""
Main script for occlusion detection project.
Orchestrates data loading, training, validation, and testing.
"""

import torch
from src.dataset import load_data, validate_images, create_datasets, create_dataloaders
from src.model import get_model, get_loss_and_optimizer
from src.train import train_model
from src.inference import validate, test
from config import IMAGE_DIR, NUM_EPOCHS, OUTPUT_PRED

def main():
    """Main execution function."""
    print("Loading and preparing data...")

    # Load data
    df_train, df_val, df_test = load_data()

    # Validate images (optional, can be slow)
    # validate_images(df_train, IMAGE_DIR, "training")
    # validate_images(df_val, IMAGE_DIR, "validation")
    # validate_images(df_test, IMAGE_DIR, "test")

    # Create datasets and dataloaders
    training_set, validation_set, test_set = create_datasets(
        df_train, df_val, df_test
    )
    training_loader, validation_loader, test_loader = create_dataloaders(
        training_set, validation_set, test_set
    )

    print("Initializing model...")

    # Initialize model
    model, device = get_model()
    loss_fn, optimizer = get_loss_and_optimizer(model)

    print(f"Using device: {device}")

    # Train model
    print(f"\nTraining for {NUM_EPOCHS} epoch(s)...")
    model = train_model(
        model, training_loader, validation_loader,
        loss_fn, optimizer, device
    )

    # Validate model
    print("\nRunning validation...")
    val_results, metric_score = validate(model, validation_loader, device)
    print(f"Validation metric: {metric_score:.6f}")

    # Test model
    print("\nRunning inference on test set...")
    test_results = test(model, test_loader, device)
    print(f"Test predictions saved to {OUTPUT_PRED}")

    print("\nDone!")

if __name__ == "__main__":
    main()