# Face Occlusion Prediction

> A PyTorch-based project for predicting facial occlusion levels from images.


# Project Structure

```bash
project/
├── crops/                        # Dataset csv
└── DataChallenge2026/
    ├── occlusion_datasets/       # Dataset
    ├── src/
    │   ├── dataset.py            # Data loading and preprocessing
    │   ├── train.py              # Training loop and optimization
    │   ├── inference.py          # Validation and test inference
    │   └── model.py              # Model architecture
    ├── config.py                 
    ├── metrics.py                # Challenge metrics
    └── main.py        
```


```bash
python main.py
```

# 1 - Pipeline

1. **Data Loading**

   * Load training, validation, and test CSV files
   * Normalize `FaceOcclusion` labels
   * Create balanced occlusion/gender groups
   * Apply stratified train/validation split

2. **Dataset & DataLoader Creation**

   * Build PyTorch datasets with augmentations
   * Use a `WeightedRandomSampler` to reduce class imbalance

3. **Model Training**

   * Train one or multiple models (`MODEL_NAME`)
   * Apply:

     * Gradient clipping
     * Learning-rate scheduling
     * Early stopping
   * Save the best checkpoint (`best_model.pth`)

4. **Validation**

   * Run ensemble validation using all trained models
   * Compute:

     * Validation MAE
     * Fairness metrics
     * Occlusion-level performance

5. **Test Inference**

   * Generate ensemble predictions on the test set
   * Save predictions to:


# 2 - Dataset

| Dataset        | Source                                 | Size    | Labels                                |
| -------------- | -------------------------------------- | ------- | ------------------------------------- |
| **Train**      | `occlusion_datasets/train.csv`         | 80,000+ | `filename`, `FaceOcclusion`, `gender` |
| **Validation** | Split from training set                | 20,000  | Same as train                         |
| **Test**       | `occlusion_datasets/test_students.csv` | Unknown | `filename` only                       |
| **Images**     | `../crops/Crop_224_5fp_100K/`          | 224×224 | `.jpg` / `.png`                       |

## Label Description

* `FaceOcclusion`:

  * Continuous value between `0` and `1`
  * `0` = no occlusion
  * `1` = fully occluded face

* `gender`:

  * `0` = female
  * `1` = male

---

# 3 - Methodology

## Model(s)

* convnext_tiny
* efficientnet_b2
* efficientnet_v2_s


## Data Augmentation

Training images are augmented using:

* Random horizontal flip
* Random rotation
* Color jitter
* Random erasing
* Image normalization

This improves robustness and generalization.


## Balanced Sampling Strategy

To reduce dataset imbalance:

* Occlusion scores are grouped into bins
* Gender and occlusion bins are combined into groups
* A `WeightedRandomSampler` is used to balance training batches

This ensures fairer learning across:

* Different occlusion levels
* Male and female samples


## Training

The training pipeline includes:

* Gradient clipping
* Learning-rate scheduling
* Validation monitoring
* Automatic checkpoint saving
* Early stopping

### Early Stopping

Training stops automatically when validation **MAE** no longer improves after a fixed number of epochs.


## Ensemble Inference

Inference supports model ensembling:

* Multiple trained models can be loaded
* Predictions are averaged
* Final predictions are clamped between `0` and `1`

This improves stability and prediction accuracy.

---

# 5 - Evaluation

The weighted error is defined as:

$$
Err = \frac{\sum_i w_i (p_i - GT_i)^2}{\sum_i w_i}
$$

with:

$$
w_i = \frac{1}{30} + GT_i
$$

$$
Score = \frac{Err_F + Err_M}{2} + \left| Err_F - Err_M \right|
$$

# 6 - Outputs

| File                   | Description                                       |
| ---------------------- | ------------------------------------------------- |
| `best_model.pth`       | Best saved model checkpoint                       |
| `test_predictions.csv` | Predicted `FaceOcclusion` values for the test set |



