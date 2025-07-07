# IJmuiden Cross-Current Forecasting Project

## Overview

This project focuses on enhancing cross-current forecasting for the Port of Amsterdam using machine learning techniques. The goal is to improve upon existing legacy neural network models by incorporating environmental measurements including wind direction, water height, flow rate, and flow direction.

## Project Goals

- **Primary Objective**: Enhance the accuracy and reliability of cross-current forecasting at the Port of Amsterdam
- **Data Integration**: Process and combine multiple environmental data sources
- **Model Development**: Implement Gaussian Process and Neural Network models for forecasting
- **Feature Engineering**: Create robust features for improved prediction accuracy

## Data Sources

The project uses the following environmental measurements:
- **Wind Direction**: Wind direction in degrees
- **Water Height**: Water height in centimeters
- **Flow Rate**: Water current speed in cm/s
- **Flow Direction**: Water current direction in degrees

## Project Structure

```
ijmuiden_cross-current/
├── data/ # Data directory (not tracked in git)
│ ├── raw/ # Raw CSV data files
│ └── processed/ # Processed and normalized data
├── notebooks/ # Jupyter notebooks for analysis
│ ├── preprocessing.ipynb # Data preprocessing and exploration
│ ├── feat_eng.ipynb # Feature engineering
│ ├── gaussian.ipynb # Gaussian Process modeling
│ └── neuralnet.ipynb # Neural Network modeling
├── src/ # Source code
│ ├── data/ # Data processing modules
│ │ ├── ingest.py # Data ingestion and preprocessing
│ │ └── features.py # Feature engineering utilities
│ ├── models/ # Model implementations
│ │ └── base.py # Base model classes
│ └── utils/ # Utility functions
│ └── data_utils.py # Data utilities
├── scripts/ # Executable scripts
│ ├── ingest_data.py # Data ingestion script
│ ├── train.py # Model training script
│ ├── predict.py # Prediction script
│ └── eval.py # Model evaluation script
├── models/ # Saved model files
├── tests/ # Unit tests
├── environment.yml # Conda environment configuration
└── README.md # Project documentation
```

## Setup Instructions

### Prerequisites

- Python 3.9+
- Conda or Miniconda

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/ijmuiden_cross-current.git
   cd ijmuiden_cross-current
   ```

2. **Create the conda environment**:
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the environment**:
   ```bash
   conda activate ijmuiden-crosscurrent
   ```

4. **Verify installation**:
   ```bash
   python -c "import pandas, numpy, torch, gpytorch; print('All packages installed successfully!')"
   ```

## Data Setup

### Data Requirements

Place your raw data files in the `data/raw/` directory. The system expects CSV files with the following structure:

- **wind_direction.csv**: Wind direction measurements
- **water_height.csv**: Water height measurements  
- **flow_rate.csv**: Water current speed measurements
- **flow_direction.csv**: Water current direction measurements

### Data Processing

1. **Run data ingestion**:
   ```bash
   python scripts/ingest_data.py
   ```
   
   This will:
   - Load and clean raw data
   - Calculate normalization parameters
   - Save processed data to `data/processed/`
   - Generate normalization parameters file

2. **Check processed data**:
   ```bash
   ls data/processed/
   ```

## Usage

### Data Preprocessing

Start with the preprocessing notebook to understand your data:

```bash
jupyter notebook notebooks/preprocessing.ipynb
```

This notebook will:
- Load and explore the raw data
- Perform data cleaning and validation
- Generate data quality reports
- Create initial visualizations

### Feature Engineering

Use the feature engineering notebook to create predictive features:

```bash
jupyter notebook notebooks/feat_eng.ipynb
```

### Model Development

#### Gaussian Process Models

```bash
jupyter notebook notebooks/gaussian.ipynb
```

The Gaussian Process implementation includes:
- Spatio-temporal kernel design
- Multi-output prediction
- Uncertainty quantification

#### Neural Network Models

```bash
jupyter notebook notebooks/neuralnet.ipynb
```

### Scripts

#### Data Ingestion
```bash
python scripts/ingest_data.py
```

#### Model Training
```bash
python scripts/train.py
```

#### Predictions
```bash
python scripts/predict.py
```

#### Model Evaluation
```bash
python scripts/eval.py
```

## Key Features

### Data Processing
- **Automated Ingestion**: Smart CSV parsing with error handling
- **Data Cleaning**: Outlier removal and missing value handling
- **Normalization**: Z-score normalization with parameter persistence
- **Type Safety**: Automatic data type conversion and validation

### Model Development
- **Gaussian Processes**: Spatio-temporal modeling with custom kernels
- **Neural Networks**: Deep learning approaches for time series
- **Feature Engineering**: Temporal and spatial feature extraction
- **Cross-validation**: Robust model evaluation

### Visualization
- **Time Series Plots**: Temporal pattern analysis
- **Correlation Analysis**: Feature relationship exploration
- **Model Diagnostics**: Performance and uncertainty visualization

## Data Format

### Input Data Structure

Raw CSV files should contain:
- `WAARNEMINGDATUM`: Date column
- `WAARNEMINGTIJD (MET/CET)`: Time column
- `NUMERIEKEWAARDE`: Measurement value
- `X`, `Y`: Spatial coordinates (optional)

### Processed Data Output

The ingestion process creates:
- Cleaned and normalized CSV files
- Normalization parameters (JSON)
- Data quality reports

## License

This project is proprietary to VORtech and the Port of Amsterdam.

## Acknowledgments

- VORtech for project supervision
- Port of Amsterdam for data and domain expertise
- Open source community for the tools and libraries used