"""
Dataset module for occlusion detection.
Handles data loading, preprocessing, and DataLoader creation.
"""

import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from config import IMAGE_DIR, TRAIN_CSV, TEST_CSV, VAL_SIZE, BATCH_SIZE, NUM_WORKERS

class OcclusionDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for occlusion detection."""

    def __init__(self, df, image_dir, training=True):
        """
        Args:
            df: DataFrame containing image filenames and labels
            image_dir: Directory containing images
            training: If True, returns labels; if False, returns only images
        """
        self.training = training
        self.image_dir = image_dir
        self.df = df
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        filename = row['filename']

        # Load image
        img = Image.open(f"{self.image_dir}/{filename}")
        X = self.transform(img)

        if self.training:
            y = row['FaceOcclusion']
            y = float(y)  # Convert to Python float for compatibility
            gender = row['gender']
            return X, y, gender, filename
        else:
            return X, filename

def load_data():
    """Load and preprocess training and test data."""
    # Load CSV files
    df_train = pd.read_csv(TRAIN_CSV, delimiter=',')
    df_test = pd.read_csv(TEST_CSV, delimiter=',')

    # Remove rows with missing values
    df_train = df_train.dropna()
    df_test = df_test.dropna()

    # Split training into train and validation
    df_val = df_train.loc[:VAL_SIZE].reset_index(drop=True)
    df_train = df_train.loc[VAL_SIZE:].reset_index(drop=True)

    return df_train, df_val, df_test

def validate_images(df, image_dir, set_name):
    """
    Validate that all images in the DataFrame can be opened.

    Args:
        df: DataFrame containing image filenames
        image_dir: Directory containing images
        set_name: Name of the dataset (for progress bar)
    """
    print(f"Validating images for {set_name}...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Validating {set_name}"):
        try:
            filename = df.loc[idx, 'filename']
            Image.open(f"{image_dir}/{filename}")
        except (ValueError, FileNotFoundError) as e:
            print(f"Error at index {idx}: {e}")

def create_datasets(df_train, df_val, df_test, image_dir=IMAGE_DIR):
    """Create dataset objects for training, validation, and test."""
    training_set = OcclusionDataset(df_train, image_dir, training=True)
    validation_set = OcclusionDataset(df_val, image_dir, training=True)
    test_set = OcclusionDataset(df_test, image_dir, training=False)

    return training_set, validation_set, test_set

def create_dataloaders(training_set, validation_set, test_set):
    """Create DataLoader objects for training, validation, and test."""
    params_train = {
        'batch_size': BATCH_SIZE,
        'shuffle': True,
        'num_workers': NUM_WORKERS
    }

    params_val = {
        'batch_size': BATCH_SIZE,
        'shuffle': False,
        'num_workers': NUM_WORKERS
    }

    training_loader = torch.utils.data.DataLoader(training_set, **params_train)
    validation_loader = torch.utils.data.DataLoader(validation_set, **params_val)
    test_loader = torch.utils.data.DataLoader(test_set, **params_val)

    return training_loader, validation_loader, test_loader