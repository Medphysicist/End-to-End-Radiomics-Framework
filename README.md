# End-to-End Radiomics Pipeline

This Streamlit application provides a complete, web-based pipeline for radiomics analysis. Users can upload DICOM medical imaging data, extract a comprehensive set of radiomic features, and perform downstream analysis to find correlations with clinical outcomes.

## ✨ Features

- **Flexible Data Input**: Upload individual DICOM files, ZIP archives, or specify a server path.
- **Automated Pre-processing**: Automatically organizes DICOM files and converts RTSTRUCT contours into NIfTI masks.
- **Customizable Feature Extraction**: Configure PyRadiomics settings, including normalization, resampling, and feature classes, directly from the UI.
- **Clinical Correlation**: Upload clinical data to merge with radiomic features.
- **Advanced Analysis**:
  - Univariate correlation analysis to find top features.
  - Multivariate feature selection using Lasso regression.
  - Interactive visualizations, including correlation heatmaps.
- **System Monitoring**: Keep an eye on server resources.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/radiomics_pipeline.git
    cd radiomics_pipeline
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

To start the Streamlit server, run the following command in your terminal:

```bash
streamlit run app.py# Radiomics_Pipeline
