# Dataset Documentation

## SDO/HMI Instrument Description
The Solar Dynamics Observatory (SDO) is a NASA mission designed to study the solar atmosphere. The Helioseismic and Magnetic Imager (HMI) is an instrument on SDO that provides continuous full-disk observations of the solar photosphere. HMI measures the line-of-sight (LOS) magnetic field and the vector magnetic field, mapping the magnetic fields that drive solar activity.

## SHARP Data Product
The Space-weather HMI Active Region Patches (SHARP) data product (hmi.sharp_cea_720s) provides automatically tracked cutouts of solar active regions. SHARP data provides both the raw magnetogram arrays and a set of derived physical parameters (e.g., total unsigned magnetic flux, magnetic shear, helicity) that summarize the magnetic complexity of the active region.

## JSOC Data Access Instructions
Data from SDO/HMI is hosted by the Joint Science Operations Center (JSOC) at Stanford University.
1. **Register**: Create an account on the JSOC website (jsoc.stanford.edu) and register your email.
2. **drms Python Package**: Use the `drms` Python library to query and download data programmatically.
3. **Query**: Construct queries using the data series name (e.g., `hmi.sharp_cea_720s`) and a time range.

## Data Format
- **FITS Files**: HMI data is typically distributed in Flexible Image Transport System (FITS) format, standard in astronomy.
- **Magnetogram Arrays**: A FITS file contains a 2D NumPy-compatible array representing the magnetic field. Positive values indicate outward-pointing fields, and negative values indicate inward-pointing fields.

## Sample Dataset
A sample, **synthetic dataset** is included in this repository under `data/sample/`. 
- **Purpose**: This dataset is purely for testing the pipeline, verifying code functionality, and demonstrating the API/UI.
- **Warning**: It does NOT contain real physics data and cannot be used for scientific modeling or accurate flare prediction.

## How to Obtain Real Data
To run the full pipeline for scientific purposes, you must download real data:
1. Install the `drms` and `sunpy` packages.
2. Modify the JSOC download script in `scripts/download_jsoc.py` (you will need to implement this or use the provided snippets).
3. Set your registered JSOC email in the environment variable `JSOC_EMAIL`.
4. Run the script to fetch FITS files for a specified list of Active Regions and dates.

## Data Statistics
- **Typical Magnetogram Ranges**: Quiet sun regions typically have field strengths around 0-50 Gauss. Active region field strengths can exceed 2000-3000 Gauss.
- **Dimensions**: SHARP cutouts vary in size depending on the active region, hence the need for resizing in the preprocessing pipeline.

## Labeling Procedure
1. Obtain the Geostationary Operational Environmental Satellite (GOES) flare catalog.
2. For each SHARP observation timestamp, check a 24-hour forward-looking window.
3. Search the GOES catalog for flares that occurred within that window.
4. Spatially match the flare location to the active region NOAA number.
5. If an M-class (>= 1e-5 W/m²) or X-class (>= 1e-4 W/m²) flare occurred, assign label 1. Otherwise, assign label 0.

## Known Dataset Issues and Biases
- **Projection Effects**: As active regions rotate toward the solar limb, LOS magnetograms suffer from foreshortening and projection effects, making magnetic measurements less accurate.
- **Instrument Noise**: HMI has known noise levels, particularly in regions of weak magnetic field.
- **Class Imbalance**: There are vastly more negative examples (no flare) than positive examples, heavily skewing the dataset.

## Reference Dataset
For standard benchmarking, we recommend the dataset compiled by Galvez et al. 2019:
> Galvez, R. et al. (2019). "A Machine-Learning Data Set Prepared from the NASA SDO Mission." *The Astrophysical Journal Supplement Series*, 242(1), 7.
This dataset provides a pre-curated, ML-ready version of SDO data.
