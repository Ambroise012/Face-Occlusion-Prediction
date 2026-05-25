"""
Dataset module for occlusion detection.
Handles data loading, preprocessing, and DataLoader creation.
"""

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from config import IMAGE_DIR, TRAIN_CSV, TEST_CSV, BATCH_SIZE, NUM_WORKERS


# =========================================================
# TRANSFORMS
# =========================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.ToTensor(),
    transforms.RandomErasing(
        p=0.5,
        scale=(0.02, 0.25),
        ratio=(0.3, 3.3),
        value='random'
    ),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class OcclusionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df,
        image_dir,
        transform,
        training=True
    ):

        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        filename = row['filename']
        path = f"{self.image_dir}/{filename}"
        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        if self.training:
            # normalized target between 0 and 1
            target = float(row['FaceOcclusion'])
            gender = row['gender']
            return image, target, gender, filename

        return image, filename

# =========================================================
# BINS FOR BALANCING (equalize)
# =========================================================

def create_occlusion_bins(df):
    bins = [0, 0.1, 0.2, 0.3, 0.4, 1.0]

    df['occ_bin'] = np.digitize(
        df['FaceOcclusion'],
        bins
    )
    return df


def create_combined_group(df):
    df['group'] = (
        df['gender'].astype(str)
        + "_"
        + df['occ_bin'].astype(str)
    )
    return df




def load_data():
    """Load and preprocess training and test data."""
    # Load CSV files
    df_train = pd.read_csv(TRAIN_CSV, delimiter=',')
    df_test = pd.read_csv(TEST_CSV, delimiter=',')

    # Remove rows with missing values
    df_train = df_train.dropna()
    df_test = df_test.dropna()

    # normalize labels between 0 and 1
    if df_train['FaceOcclusion'].max() > 1.0:

        df_train['FaceOcclusion'] /= 100.0

    # create balancing bins
    df_train = create_occlusion_bins(df_train)

    # combine gender + occlusion
    df_train = create_combined_group(df_train)

    # stratified split
    train_df, val_df = train_test_split(
        df_train,
        test_size=0.2,
        stratify=df_train['group'],
        random_state=42
    )

    train_df = train_df.reset_index(drop=True)

    val_df = val_df.reset_index(drop=True)

    return train_df, val_df, df_test


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

def create_datasets(
    train_df,
    val_df,
    test_df
):

    training_set = OcclusionDataset(
        train_df,
        IMAGE_DIR,
        transform=train_transform,
        training=True
    )

    validation_set = OcclusionDataset(
        val_df,
        IMAGE_DIR,
        transform=val_transform,
        training=True
    )

    test_set = OcclusionDataset(
        test_df,
        IMAGE_DIR,
        transform=val_transform,
        training=False
    )

    return (
        training_set,
        validation_set,
        test_set
    )


def create_dataloaders(
    training_set,
    validation_set,
    test_set,
    train_df
):

    sampler = create_weighted_sampler(
        train_df
    )

    training_loader = DataLoader(
        training_set,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    validation_loader = DataLoader(
        validation_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    return (
        training_loader,
        validation_loader,
        test_loader
    )


# =========================================================
# WEIGHTED SAMPLER
# =========================================================

# def create_weighted_sampler(df):

#     weights = []

#     for _, row in df.iterrows():
#         target = row['FaceOcclusion']
#         gender = row['gender']

#         # OCCLUSION WEIGHT
#         if target > 0.5:
#             occ_weight = 25
#         elif target > 0.4:
#             occ_weight = 15
#         elif target > 0.3:
#             occ_weight = 8
#         elif target > 0.2:
#             occ_weight = 4
#         elif target > 0.1:
#             occ_weight = 2
#         else:
#             occ_weight = 1

#         # GENDER BALANCING
#         gender_weight = 1.8 if gender == 0.0 else 1.0

#         # FINAL WEIGHT
#         final_weight = (
#             occ_weight * gender_weight
#         )

#         weights.append(final_weight)

#     weights = torch.DoubleTensor(weights)

#     sampler = WeightedRandomSampler(
#         weights,
#         num_samples=len(weights),
#         replacement=True
#     )

#     return sampler


def create_weighted_sampler(df):
    

    group_counts = (
        df['group']
        .value_counts()
    )

    group_weights = 1.0 / group_counts

    sample_weights = (
        df['group']
        .map(group_weights)
        .values
    )

    sample_weights = torch.DoubleTensor(
        sample_weights
    )

    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler