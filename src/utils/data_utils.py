# The data utils module contains functions for loading, preprocessing, and splitting the data.
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class CrossCurrentDataset(Dataset):
    """Custom Dataset for cross-current time series data."""
    
    def __init__(self, data, targets, sequence_length=24):
        """
        Initialize the dataset.
        
        Args:
            data: Input features (n_samples, n_features)
            targets: Target values (n_samples, n_targets)
            sequence_length: Number of time steps to use as input
        """
        self.data = torch.FloatTensor(data)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length
        
    def __getitem__(self, idx):
        # Get sequence of input data
        x = self.data[idx:idx + self.sequence_length]
        # Get target (next value after sequence)
        y = self.targets[idx + self.sequence_length]
        return x, y


def load_data(data_path):
    # Define paths
    processed_data_dir = Path(data_path)

    # List processed files
    processed_files = list(processed_data_dir.glob('*_processed.csv'))
    print(f"Found {len(processed_files)} processed data files:")
    for file in processed_files:
        print(f"  - {file.name}")

    # Load data into a dictionary
    data_dict = {}
    for file_path in processed_files:
        data_type = file_path.stem.replace('_processed', '')
        print(f"\nLoading {data_type}...")
        
        try:
            df = pd.read_csv(file_path)
            # Convert datetime column
            df['datetime'] = pd.to_datetime(df['datetime'])
            data_dict[data_type] = df
            print(f"  ✓ Loaded {len(df)} records")
            print(f"  ✓ Date range: {df['datetime'].min()} to {df['datetime'].max()}")
            print(f"  ✓ Columns: {list(df.columns)}")
        except Exception as e:
            print(f"  ✗ Error loading {data_type}: {e}")
    
    if len(data_dict.items()) == 0:
        print("No data loaded")
        return None
    else:
        print(f"\nSuccessfully loaded {len(data_dict)} datasets")
    return data_dict

def prepare_current_direction_features(data_dict, inference=False):
    """
    Prepare features and targets for current direction prediction.
    
    Args:
        data_dict: Dictionary of datasets
        sequence_length: Number of time steps to use as input
        
    Returns:
        Tuple of (features, targets, feature_columns, target_column)
    """
    print("Preparing features for current direction prediction...")
    
    # Merge all datasets on datetime
    merged_data = None
    for data_type, df in data_dict.items():
        df_copy = df.copy()
        # Rename value to avoid conflicts
        df_copy = df_copy.rename(columns={'value': f'{data_type}'})
        
        if merged_data is None:
            merged_data = df_copy[['datetime', 'datetime_unix', f'{data_type}']]
        else:
            merged_data = merged_data.merge(
                df_copy[['datetime', f'{data_type}']], 
                on='datetime', how='inner'
            )
    
    # Sort by datetime
    merged_data = merged_data.sort_values('datetime').reset_index(drop=True)
    datetimes = merged_data['datetime'].values
    
    # Select feature columns (all variables except current_direction)
    feature_columns = [col for col in merged_data.columns]
    features = merged_data[feature_columns].values

    if inference:
        feature_columns.remove('datetime')
        features = merged_data[feature_columns].values
        return features, feature_columns, datetimes

    target_column = 'cross_current'
    
    # Remove target from features
    if target_column in feature_columns:
        feature_columns.remove(target_column)
        feature_columns.remove('datetime')
    
    # Prepare features and targets
    features = merged_data[feature_columns].values
    targets = merged_data[target_column].values
    
    print(f"  ✓ Features shape: {features.shape}")
    print(f"  ✓ Targets shape: {targets.shape}")
    print(f"  ✓ Feature columns: {feature_columns}")
    print(f"  ✓ Target column: {target_column}")
    
    return features, targets, feature_columns, target_column, datetimes

def create_datasets(features, targets, train_split=0.7, val_split=0.15, sequence_length=24):
    """
    Create train, validation, and test datasets.
    
    Args:
        features: Input features
        targets: Target values
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        sequence_length: Number of time steps for sequences
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    print("Creating datasets...")
    
    # Calculate split indices
    n_samples = len(features) - sequence_length
    train_end = int(n_samples * train_split)
    val_end = int(n_samples * (train_split + val_split))
    
    # Create datasets
    train_dataset = CrossCurrentDataset(
        features[:train_end + sequence_length], 
        targets[:train_end + sequence_length], 
        sequence_length
    )
    
    val_dataset = CrossCurrentDataset(
        features[train_end:val_end + sequence_length], 
        targets[train_end:val_end + sequence_length], 
        sequence_length
    )
    
    test_dataset = CrossCurrentDataset(
        features[val_end:], 
        targets[val_end:], 
        sequence_length
    )
    
    print(f"  ✓ Train samples: {len(train_dataset)}")
    print(f"  ✓ Validation samples: {len(val_dataset)}")
    print(f"  ✓ Test samples: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset

def plot_predictions(predictions, actuals=None, datetimes=None, title="Cross Current Predictions", 
                    mode="inference", max_time_points=200):
    """
    Unified plotting function for both training evaluation and inference scenarios.
    
    Args:
        predictions: Array of predictions
        actuals: Array of actual values (optional, for evaluation mode)
        datetimes: Array of datetime values (optional)
        title: Title for the plot
        mode: Either "evaluation" (with actuals) or "inference" (predictions only)
        max_time_points: Maximum number of points to show in time series plots
    """
    if mode == "evaluation" and actuals is not None:
        # Evaluation mode - show actual vs predicted comparisons
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Scatter plot
        axes[0, 0].scatter(actuals, predictions, alpha=0.5)
        axes[0, 0].plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Cross Current')
        axes[0, 0].set_ylabel('Predicted Cross Current')
        axes[0, 0].set_title(f'{title} - Scatter Plot')
        axes[0, 0].grid(True)
        
        # Time series plot (limited points)
        axes[0, 1].plot(datetimes[:max_time_points], 
                       actuals[:max_time_points], label='Actual', alpha=0.7)
        axes[0, 1].plot(datetimes[:max_time_points], 
                       predictions[:max_time_points], label='Predicted', alpha=0.7)
        axes[0, 1].set_xlabel('Datetime')
        axes[0, 1].set_ylabel('Cross Current')
        axes[0, 1].set_title(f'{title} - Time Series')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Residuals plot
        residuals = actuals - predictions
        axes[1, 0].scatter(predictions, residuals, alpha=0.5)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Predicted Cross Current')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Residuals Plot')
        axes[1, 0].grid(True)
        
        # Residuals histogram
        axes[1, 1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Residuals Distribution')
        axes[1, 1].grid(True)
        
    else:
        # Inference mode - show predictions only
        fig, axes = plt.subplots(2, 2, figsize=(15, 8))
        
        # Time series plot
        if datetimes is not None:
            axes[0, 0].plot(datetimes, predictions, linewidth=1, alpha=0.8)
            axes[0, 0].set_xlabel('Datetime')
        else:
            axes[0, 0].plot(predictions, linewidth=1, alpha=0.8)
            axes[0, 0].set_xlabel('Time Steps')
        axes[0, 0].set_ylabel('Cross Current')
        axes[0, 0].set_title(f'{title} - Time Series')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Histogram of predictions
        axes[0, 1].hist(predictions, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Cross Current')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'{title} - Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Rolling statistics
        window_size = min(100, len(predictions) // 10)
        if window_size > 1:
            rolling_mean = pd.Series(predictions).rolling(window=window_size).mean()
            rolling_std = pd.Series(predictions).rolling(window=window_size).std()
            
            axes[1, 0].plot(predictions, alpha=0.5, label='Predictions', linewidth=0.5)
            axes[1, 0].plot(rolling_mean, label=f'Rolling Mean ({window_size} steps)', linewidth=2)
            axes[1, 0].fill_between(range(len(predictions)), 
                                  rolling_mean - rolling_std, 
                                  rolling_mean + rolling_std, 
                                  alpha=0.3, label='±1 Std Dev')
            axes[1, 0].set_title(f'{title} - Rolling Statistics')
            axes[1, 0].set_xlabel('Time Steps')
            axes[1, 0].set_ylabel('Cross Current')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Not enough data for rolling stats', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title(f'{title} - Rolling Statistics')
        
        # Summary statistics
        stats_text = f"""
        Statistics:
        Count: {len(predictions):,}
        Mean: {predictions.mean():.3f}
        Std: {predictions.std():.3f}
        Min: {predictions.min():.3f}
        Max: {predictions.max():.3f}
        Range: {predictions.max() - predictions.min():.3f}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        axes[1, 1].set_title(f'{title} - Summary Statistics')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()