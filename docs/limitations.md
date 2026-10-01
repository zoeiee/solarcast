# Limitations

While SolarCast provides a modern machine learning architecture for solar flare prediction, it is essential to acknowledge its scientific and operational limitations.

## Data Limitations
- **Synthetic Sample Data**: The dataset included in this repository (`data/sample`) is purely synthetic and generated for software testing. Models trained on this data hold no scientific validity and cannot predict real solar flares.
- **Simplified Features**: The model relies solely on raw magnetogram images. It does not utilize the highly predictive, physically derived SHARP parameters (e.g., total unsigned flux, current helicity) calculated by the JSOC pipeline, which are known to improve model performance.

## Modeling Limitations
- **Image-Only Approach**: By using single images (2D spatial data), the model misses the crucial temporal evolution of the active region. The rate of flux emergence and changes in magnetic shear over time are strong predictors of flaring activity that a single static snapshot cannot capture.
- **Binary Classification**: Formulating the problem as a binary classification (M/X vs. A/B/C/Quiet) loses severity information. An X9 flare is vastly more impactful than an M1 flare, but both receive the same label.
- **Fixed Prediction Window**: The model uses a fixed 24-hour forward window. It cannot specify *when* within that 24-hour period the flare will occur.
- **No Ensemble Methods**: State-of-the-art forecasting often relies on ensembles of different models to quantify uncertainty. SolarCast currently implements a single deterministic CNN.
- **Calibration**: The output probabilities are not rigorously calibrated. While a threshold can be tuned, a predicted probability of 0.8 does not strictly equate to an 80% real-world chance of occurrence without proper Platt scaling or Isotonic regression.

## Interpretability
- **Grad-CAM Limitations**: While Grad-CAM highlights regions the CNN focuses on (e.g., strong magnetic gradients), it shows correlation, not causation. It cannot explain the physical mechanism the model learned, and it can sometimes highlight arbitrary features due to network artifacts.

## Operational Readiness
- **Not for Operational Use**: SolarCast is a research and educational prototype. It has not undergone the rigorous operational validation required by space weather agencies (like NOAA's SWPC). It should not be used to make decisions regarding satellite operations, power grid management, or human spaceflight safety.

## Future Work Needed
To advance this project toward state-of-the-art scientific forecasting, future work must include:
1. **Temporal Modeling**: Implementing Recurrent Neural Networks (LSTMs) or 3D CNNs to process a time-series of magnetograms.
2. **Multi-Modal Data**: Integrating SHARP scalar parameters and extreme ultraviolet (EUV) images from SDO/AIA alongside the magnetograms.
3. **Operational Validation**: Testing the model strictly chronologically on years of real SDO data to simulate true operational forecasting conditions.
