# IJmuiden Cross-Current Forecasting Project

## Overview

This project focuses on enhancing cross-current forecasting for the Port of Amsterdam using machine learning techniques. The goal is to improve upon existing legacy neural network models by incorporating Acoustic Doppler Current Profiler (ADCP) measurements and freshwater outflow estimates.

## Project Goals

- **Primary Objective**: Enhance the accuracy and reliability of cross-current forecasting at the Port of Amsterdam
- **Data Integration**: Combine ADCP measurements with freshwater outflow estimates
- **Model Improvement**: Develop and validate improved forecasting models
- **Operational Deployment**: Create models suitable for real-time operational use

## Data Sources

- **ADCP Measurements**: Acoustic Doppler Current Profiler data providing current velocity profiles
- **Freshwater Outflow Estimates**: River discharge and freshwater flow data
- **Historical Data**: Legacy forecasting data and validation records

## Project Structure

```
ijmuiden_cross-current/
├── data/                    # Raw and processed data files
│   ├── 20250702_033.csv    # Main ADCP dataset
│   ├── 20250702_034.csv    # Additional dataset
│   └── *.zip               # Compressed data files
├── notebooks/              # Jupyter notebooks for analysis
│   ├── exploration_01.ipynb # Initial data exploration
│   └── baseline_model.ipynb # Baseline model development
├── src/                    # Source code
│   ├── data_utils.py       # Data processing utilities
│   └── main.py            # Main application entry point
├── requirements.txt        # Python package requirements
├── environment.yml         # Conda environment configuration
└── README.md              # Project documentation
```

## Setup Instructions

### Option 1: Using Conda (Recommended)

1. **Create the conda environment**:
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the environment**:
   ```bash
   conda activate ijmuiden-crosscurrent
   ```

3. **Verify installation**:
   ```bash
   python -c "import pandas, numpy, matplotlib, seaborn; print('All packages installed successfully!')"
   ```

### Option 2: Using pip

1. **Create a virtual environment**:
   ```bash
   python -m venv ijmuiden-env
   source ijmuiden-env/bin/activate  # On Windows: ijmuiden-env\Scripts\activate
   ```

2. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

## Getting Started

1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open the exploration notebook**:
   - Navigate to `notebooks/exploration_01.ipynb`
   - Run the cells to explore your data

3. **Data Exploration**:
   - The notebook will automatically load and analyze your data
   - Visualize distributions, correlations, and time series patterns
   - Identify key features for forecasting

## Key Features

- **Automated Data Loading**: Smart loading with fallback options for large datasets
- **Data Quality Analysis**: Missing value detection and data type analysis
- **Visualization**: Comprehensive plotting of distributions, correlations, and time series
- **Time Series Analysis**: Automatic datetime handling and temporal pattern detection

## Next Steps

1. **Data Understanding**: Run the exploration notebook to understand your data structure
2. **Feature Engineering**: Identify and create relevant features for forecasting
3. **Model Development**: Build baseline and advanced forecasting models
4. **Validation**: Implement cross-validation and performance metrics
5. **Deployment**: Prepare models for operational use

## Contributing

This is an internship project at VORtech. For questions or contributions, please contact the project supervisor.

## License

This project is proprietary to VORtech and the Port of Amsterdam.
