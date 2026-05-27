"""
Training module for occlusion detection.
Handles the training loop and model optimization.
"""

import torch
from tqdm import tqdm


class EarlyStoppinng:
    def __init__(
            self,
            patience=0,
            min_delta=0.0005
        ):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, metric):

        # lower MAE is better
        score = -metric

        if self.best_score is None:
            self.best_score = score
            return

        # no significant improvement
        if score < self.best_score + self.min_delta:
            self.counter += 1
            print(
                f"EarlyStopping "
                f"{self.counter}/{self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0 


def train_model(
        model, 
        training_loader, 
        validation_loader, 
        loss_fn,
        optimizer, 
        scheduler, 
        device, 
        num_epochs, 
        patience=5, 
        min_delta=0.0005
    ):
    best_val_mae, early_stop_counter = float("inf"), 0
    history = {"train_loss": [], "train_mae": [], "val_loss": [], "val_mae": []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # TRAIN
        model.train()
        running_train_loss, running_train_mae = 0.0, 0.0
        train_pbar = tqdm(training_loader, total=len(training_loader))

        for batch_idx, (images, targets, gender, filename) in enumerate(train_pbar):
            images = images.to(device, non_blocking=True)
            targets = targets.float().to(device, non_blocking=True)

            optimizer.zero_grad()
            predictions = model(images)
            loss = loss_fn(predictions, targets)

            if torch.isnan(loss):
                print(f"\nNaN LOSS DETECTED\nBatch index: {batch_idx}\nFilename: {filename}\nTargets: {targets}\nPredictions: {predictions}")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            mae = torch.mean(torch.abs(predictions - targets))
            running_train_loss += loss.item()
            running_train_mae += mae.item()

            train_pbar.set_description(f"Train Loss: {loss.item():.4f} | Train MAE: {mae.item()*100:.2f}%")

        epoch_train_loss = running_train_loss / len(training_loader)
        epoch_train_mae = running_train_mae / len(training_loader)

        # VALIDATION
        model.eval()
        running_val_loss, running_val_mae = 0.0, 0.0

        with torch.no_grad():
            val_pbar = tqdm(validation_loader, total=len(validation_loader))

            for images, targets, gender, filename in val_pbar:
                images = images.to(device, non_blocking=True)
                targets = targets.float().to(device, non_blocking=True)

                predictions = model(images)
                loss = loss_fn(predictions, targets)
                mae = torch.mean(torch.abs(predictions - targets))

                running_val_loss += loss.item()
                running_val_mae += mae.item()

                val_pbar.set_description(f"Val Loss: {loss.item():.4f} | Val MAE: {mae.item()*100:.2f}%")

        epoch_val_loss = running_val_loss / len(validation_loader)
        epoch_val_mae = running_val_mae / len(validation_loader)

        # SCHEDULER
        scheduler.step(epoch_val_mae) if scheduler.__class__.__name__ == "ReduceLROnPlateau" else scheduler.step()

        # HISTORY
        history["train_loss"].append(epoch_train_loss)
        history["train_mae"].append(epoch_train_mae)
        history["val_loss"].append(epoch_val_loss)
        history["val_mae"].append(epoch_val_mae)

        current_lr = optimizer.param_groups[0]['lr']

        print("\n--------------------------------")
        print(f"Train Loss: {epoch_train_loss:.4f}")
        print(f"Train MAE: {epoch_train_mae*100:.2f}%")
        print(f"Val Loss: {epoch_val_loss:.4f}")
        print(f"Val MAE: {epoch_val_mae*100:.2f}%")
        print(f"Learning Rate: {current_lr:.7f}")
        print("--------------------------------")

        # SAVE BEST MODEL
        improved = (best_val_mae - epoch_val_mae) > min_delta

        if improved:
            best_val_mae, early_stop_counter = epoch_val_mae, 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': epoch_val_mae,
            }, "best_model.pth")

            print(f"\nBest model saved (Val MAE: {epoch_val_mae*100:.2f}%)")

        else:
            early_stop_counter += 1
            print(f"\nEarlyStopping: {early_stop_counter}/{patience}")

        if early_stop_counter >= patience:
            print("\nEARLY STOPPING")
            break

    # LOAD BEST MODEL
    print("\nLoading best model...")
    checkpoint = torch.load("best_model.pth", map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Best Validation MAE: {checkpoint['val_mae']*100:.2f}%")
    print("\nTraining completed.")

    return model, history