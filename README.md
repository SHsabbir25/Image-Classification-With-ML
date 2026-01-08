# Geospatial LULC Classification of Dhaka (2020) using Sentinel-2 & Random Forest

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Sentinel-2](https://img.shields.io/badge/Data-Sentinel--2_SR-red.svg)
![Random Forest](https://img.shields.io/badge/Algorithm-Random_Forest-brightgreen.svg)
![Rasterio](https://img.shields.io/badge/Library-Rasterio-green.svg)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)
![GeoTIFF](https://img.shields.io/badge/Format-GeoTIFF-blue.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

This repository presents an end-to-end machine learning pipeline for **Land Use Land Cover (LULC)** mapping of the Dhaka Metropolitan area for the year 2020. By leveraging **Sentinel-2 Surface Reflectance (SR)** data and an optimized **Random Forest (RF)** classifier, this project achieves high-precision mapping of complex urban environments.

The methodology focuses on spectral feature engineering—integrating raw multispectral bands with derived biophysical indices (NDVI, MNDWI, NDBI) to maximize class separability between spectrally similar features like bare soil and high-density built-up areas.

---

## 🛰️ Spectral Data Engineering

### 1. The Sentinel-2 Advantage
While Landsat is a standard for historical analysis, **Sentinel-2 (Level-2A)** was selected for this study due to:
*   **Spatial Granularity:** 10m to 20m resolution allows for the detection of small-scale urban features and narrow riparian corridors.
*   **SWIR Sensitivity:** The Short-Wave Infrared bands (B11, B12) are critical for differentiating between moisture-stressed vegetation, bare land, and man-made impervious surfaces.

### 2. Feature Stack & Index Logic
To enhance the model's dimensionality, we constructed a 9-band input stack. The addition of spectral indices converts raw reflectance into physical proxies:

| Feature | Type | Spectral Logic |
| :--- | :--- | :--- |
| **B2, B3, B4, B8** | VNIR Bands | Fundamental for visible color and Chlorophyll-a reflectance. |
| **B11, B12** | SWIR Bands | Key for identifying structural materials (concrete/bitumen) and soil moisture. |
| **NDVI** | Vegetation Index | `(NIR - Red) / (NIR + Red)` - Quantifies photosynthetic activity. |
| **MNDWI** | Water Index | `(Green - SWIR) / (Green + SWIR)` - Enhances open water while suppressing built-up noise. |
| **NDBI** | Built-up Index | `(SWIR - NIR) / (SWIR + NIR)` - Highlights impervious surfaces and urban density. |

---

## 🛠️ Methodology & Technical Workflow

### Phase 1: Harmonization & Pre-processing
*   **Radiometric Scaling:** Applied a scale factor of 0.0001 to convert Digital Numbers (DN) to Bottom-of-Atmosphere (BOA) reflectance.
*   **Spatial Alignment:** Automated reprojection of all vector samples and raster grids to **WGS 84 (EPSG:4326)** to ensure sub-pixel alignment during training data extraction.
*   **NoData Handling:** Implemented a robust masking system using `-1.0` as a FillValue to ensure zero-value pixels do not skew the statistical distribution.

### Phase 2: Feature Extraction
*   Used a **Centroid-based Sampling** approach to extract spectral signatures from multi-polygon shapefiles.
*   Classes labeled: `Vegetation (1)`, `Built-up (2)`, `Bareland (3)`, and `Waterbody (4)`.

### Phase 3: Machine Learning Rigor
*   **Classifier:** Random Forest (300 Trees).
*   **Stratified K-Fold Cross-Validation:** The model uses a 5-fold stratified split to ensure each training fold represents the class distribution of the entire study area.
*   **State Optimization:** Iterated through 100 random states ("epochs") to capture the most stable model architecture, saving the best-performing iteration via `joblib`.

### Phase 4: Spatial Prediction & Output
*   The trained model was deployed across the full raster extent.
*   The final output is a single-band GeoTIFF with high-compression (DEFLATE) to optimize storage without losing spatial precision.

---

## 📊 Performance Evaluation

To ensure the reliability of the 2020 LULC map, the model’s predictive power was rigorously assessed using a **Stratified 5-Fold Cross-Validation** approach across 100 iterations. This ensures that the results are not just artifacts of a specific data split but are statistically robust.

### 1. Primary Metrics
The following metrics are automatically computed and exported to the `/Results` directory:

*   **Overall Accuracy (OA):** Represents the total percentage of pixels correctly classified. 
*   **Kappa Coefficient:** Measures the agreement between the classification map and ground truth, accounting for the possibility of agreement occurring by chance.
*   **Precision (User's Accuracy):** The probability that a pixel classified into a category actually represents that category on the ground.
*   **Recall (Producer's Accuracy):** How well a certain land cover type is detected by the model (e.g., what percentage of actual water bodies were correctly identified?).
*   **F1-Score:** The harmonic mean of precision and recall, providing a balanced assessment for classes with unequal sample sizes (e.g., Waterbodies vs. Built-up).

### 2. Confusion Matrix Analysis
A visual heatmap of the Confusion Matrix is generated to identify **inter-class confusion**. 
*   **Built-up vs. Bareland:** These classes often show spectral overlap. The inclusion of **NDBI** and **SWIR bands** in this project significantly minimizes these errors.
*   **Vegetation vs. Built-up:** Using **NDVI** ensures clear separation of pervious vs. impervious surfaces.

---

## 🚀 Installation & Usage

Follow these steps to set up the environment and execute the LULC classification pipeline.

### 📋 Prerequisites
Geospatial libraries like `GDAL` and `Rasterio` can be complex to install due to C++ dependencies. It is **highly recommended** to use **Conda/Mamba** for environment management.

*   Python 3.9+
*   Anaconda or Miniconda installed

### 🛠️ Step-by-Step Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/Dhaka-LULC-RF-2020.git
    cd Dhaka-LULC-RF-2020
    ```

2.  **Create a Virtual Environment:**
    ```bash
    # Create the environment with essential geospatial dependencies
    conda create -n lulc_env python=3.9 gdal -c conda-forge
    conda activate lulc_env
    ```

3.  **Install Required Libraries:**
    ```bash
    pip install rasterio geopandas scikit-learn pandas numpy matplotlib seaborn joblib
    ```

### Running the Model

1.  **Data Preparation:** 
    *   Place your input Sentinel-2 GeoTIFF in the `F:/Springer/Random Forest Model/Data/` directory (or update the paths in the script).
    *   Ensure your training shapefiles (.shp) are in the specified subfolders.

2.  **Execute the Script:**
    Run the main Python file to process the imagery, train the model, and generate the output:
    ```bash
    python lulc_rf_model_2020.py
    ```

3.  **Outputs:**
    *   **Classified Map:** `.../Classified_Output/Sentinel2_Classified_2020.tif`
    *   **Validation Plots:** Check the `Results/` folder for `confusion_matrix.png` and `accuracy_plot.png`.

## 📁 Repository Structure
```text
├── Data/                 # Original Sentinel-2 TIFs
├── Data 2020/            # Input Training Shapefiles
├── Reproject/            # Harmonized EPSG:4326 Data
├── Training_CSV/         # Extracted Spectral Signatures
├── Results/              # Confusion Matrices & Accuracy Logs
└── Code                  # Main processing script


