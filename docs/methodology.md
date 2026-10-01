# Methodology

## Problem Formulation
The problem is formulated as a binary classification task: predicting whether an M-class or X-class solar flare will occur within a 24-hour prediction window following the observation time of an active region magnetogram. 

## Data Source
The primary data source is the Helioseismic and Magnetic Imager (HMI) aboard the Solar Dynamics Observatory (SDO). We use line-of-sight (LOS) magnetograms that measure the magnetic field strength of the solar photosphere.

## Preprocessing Pipeline
1. **Clipping**: Magnetogram pixel values are clipped to a specific range (e.g., -3000 to +3000 Gauss) to remove outliers and instrument artifacts.
2. **Normalization**: The clipped values are normalized to a standard scale (e.g., [-1, 1] or [0, 1]) to facilitate neural network training.
3. **Resizing**: The cropped active region images are resized to a fixed dimension (e.g., 224x224 pixels) to be compatible with standard CNN architectures.

## Label Construction
Labels are constructed using the GOES X-ray flare catalog. We associate flares with specific active regions using NOAA active region numbers. If an AR produces an M or X class flare within 24 hours of the magnetogram observation, the observation is labeled as positive (1). Otherwise, it is labeled as negative (0).

## Train/Val/Test Splitting Strategy
To ensure robust evaluation and prevent data leakage, we use an **active-region-based splitting strategy**.
- The dataset is split based on unique NOAA Active Region (AR) numbers rather than randomly shuffling individual images.
- All observations from a specific AR are placed entirely into either the training, validation, or test set.

### Leakage Risks Explained
- **Temporal Leakage**: Taking images of the same AR 12 minutes apart. If image A is in train and image B is in test, they are nearly identical. The model can achieve high test scores by just memorizing training images.
- **Active-Region Leakage**: If an AR evolves over a week, and days 1-4 are in train and days 5-7 are in test, the model might learn the specific morphology of that AR instead of general flare-predictive features. AR-based splitting prevents this.

## Baseline Model
A Random Forest classifier serves as our baseline. Instead of raw pixels, it uses aggregated statistical features extracted from the magnetograms (e.g., total unsigned flux, mean gradient, maximum field strength, polarity inversion line length proxies).

## CNN Architecture
Our primary model is **SolarCNN**, a Convolutional Neural Network utilizing residual blocks.
- **Input**: 1-channel magnetogram image (224x224).
- **Residual Blocks**: Incorporates skip connections to allow deeper networks without suffering from vanishing gradients. Each block contains convolutional layers, Batch Normalization, and ReLU activation.
- **Output**: A fully connected layer with a sigmoid activation function outputting the probability of a flare.

## Loss Function
To address the severe class imbalance (M/X flares are rare), we use **Weighted Binary Cross-Entropy (BCE) Loss**. The positive class is given a higher weight inversely proportional to its frequency in the training set. Optionally, **Focal Loss** can be used to further focus training on hard-to-classify examples.

## Training Procedure
- **Optimizer**: Adam optimizer.
- **Learning Rate Schedule**: Cosine Annealing learning rate schedule.
- **Regularization**: Weight decay and Dropout.
- **Early Stopping**: Training halts if the validation loss does not improve for a set number of epochs to prevent overfitting.

## Evaluation Metrics
Given the class imbalance, standard accuracy is inadequate. We use:
- **Precision**: TP / (TP + FP)
- **Recall (Sensitivity)**: TP / (TP + FN)
- **F1 Score**: Harmonic mean of Precision and Recall.
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve.
- **PR-AUC**: Area under the Precision-Recall curve.
- **True Skill Statistic (TSS)**: Balances recall of both positive and negative classes.
- **Heidke Skill Score (HSS)**: Measures improvement over random chance.

### Formulas for TSS and HSS
Let TP = True Positives, TN = True Negatives, FP = False Positives, FN = False Negatives.
N = TP + TN + FP + FN

**TSS (True Skill Statistic)**:
TSS = (TP / (TP + FN)) - (FP / (FP + TN))
*Range: [-1, 1]. 1 is perfect, 0 is random, -1 is perfectly wrong. It is independent of class imbalance.*

**HSS (Heidke Skill Score)**:
HSS = (2 * (TP * TN - FP * FN)) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))
*Range: [-1, 1]. Measures the fractional improvement of the forecast over a random forecast based on climatology.*

## Explainability
To interpret the model's predictions, we use **Grad-CAM (Gradient-weighted Class Activation Mapping)**. Grad-CAM highlights the regions of the magnetogram that contributed most strongly to the model's decision, allowing us to verify if the model is focusing on physically relevant features (like strong magnetic gradients or delta configurations) rather than artifacts.
