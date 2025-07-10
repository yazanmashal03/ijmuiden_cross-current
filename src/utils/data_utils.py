# The data utils module contains functions for loading, preprocessing, and splitting the data.
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

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
    
    return features, targets, feature_columns, target_column

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
