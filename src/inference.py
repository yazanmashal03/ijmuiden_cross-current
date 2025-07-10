from utils.data_utils import *
from models_architecture.lstm import *
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np

class InferenceDataset(Dataset):
    """Custom Dataset for inference (no targets needed)."""
    
    def __init__(self, data, sequence_length=24):
        """
        Initialize the dataset.
        
        Args:
            data: Input features (n_samples, n_features)
            sequence_length: Number of time steps to use as input
        """
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length
        
    def __getitem__(self, idx):
        # Get sequence of input data
        x = self.data[idx:idx + self.sequence_length]
        return x

def load_normalization_params(norm_params_path):
    """Load normalization parameters from JSON file."""
    with open(norm_params_path, 'r') as f:
        return json.load(f)

def denormalize_predictions(predictions, norm_params):
    """Denormalize predictions using saved normalization parameters."""
    mean_val = norm_params['cross_current']['mean']['value']
    std_val = norm_params['cross_current']['std']['value']
    
    return predictions * std_val + mean_val

def plot_predictions(datetimes, predictions, title="Cross Current Predictions"):
    """
    Plot the predictions over time.
    
    Args:
        predictions: Array of predictions
        title: Title for the plot
    """
    plt.figure(figsize=(15, 8))
    
    # Time series plot
    plt.subplot(2, 2, 1)
    plt.plot(datetimes, predictions, linewidth=1, alpha=0.8)
    plt.title(f'{title} - Time Series')
    plt.xlabel('Datetime')
    plt.ylabel('Cross Current')
    plt.grid(True, alpha=0.3)
    
    # Histogram of predictions
    plt.subplot(2, 2, 2)
    plt.hist(predictions, bins=50, alpha=0.7, edgecolor='black')
    plt.title(f'{title} - Distribution')
    plt.xlabel('Cross Current')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Rolling statistics
    plt.subplot(2, 2, 3)
    window_size = min(100, len(predictions) // 10)
    if window_size > 1:
        rolling_mean = pd.Series(predictions).rolling(window=window_size).mean()
        rolling_std = pd.Series(predictions).rolling(window=window_size).std()
        
        plt.plot(predictions, alpha=0.5, label='Predictions', linewidth=0.5)
        plt.plot(rolling_mean, label=f'Rolling Mean ({window_size} steps)', linewidth=2)
        plt.fill_between(range(len(predictions)), 
                        rolling_mean - rolling_std, 
                        rolling_mean + rolling_std, 
                        alpha=0.3, label='±1 Std Dev')
        plt.title(f'{title} - Rolling Statistics')
        plt.xlabel('Time Steps')
        plt.ylabel('Cross Current')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Not enough data for rolling stats', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(f'{title} - Rolling Statistics')
    
    # Summary statistics
    plt.subplot(2, 2, 4)
    stats_text = f"""
    Statistics:
    Count: {len(predictions):,}
    Mean: {predictions.mean():.3f}
    Std: {predictions.std():.3f}
    Min: {predictions.min():.3f}
    Max: {predictions.max():.3f}
    Range: {predictions.max() - predictions.min():.3f}
    """
    plt.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    plt.title(f'{title} - Summary Statistics')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def predict_cross_current(features, feature_columns, model, device, norm_params, sequence_length=24):
    """
    Make predictions on new data.
    
    Args:
        features: Input features (n_samples, n_features)
        feature_columns: List of feature column names
        model: Trained PyTorch model
        device: Device to run inference on
        norm_params: Normalization parameters
        sequence_length: Number of time steps for sequences
        
    Returns:
        predictions: Denormalized predictions
        timestamps: Corresponding timestamps
    """

    # Create inference dataset
    inference_dataset = InferenceDataset(features, sequence_length)
    inference_loader = DataLoader(inference_dataset, batch_size=64, shuffle=False)
    
    # Make predictions
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch_x in inference_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            predictions.extend(outputs.cpu().numpy().flatten())
    
    predictions = np.array(predictions)
    
    # Denormalize predictions
    denormalized_predictions = denormalize_predictions(predictions, norm_params)
    
    return denormalized_predictions

def main():
    # Initialize variables
    sequence_length = 24

    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load normalization parameters
    norm_params_path = 'data/processed/normalization_params.json'
    norm_params = load_normalization_params(norm_params_path)
    print("✓ Loaded normalization parameters")

    # Load data
    data_path = 'data/test'
    data_dict = load_data(data_path)

    # Prepare features
    features, feature_columns, datetimes = prepare_current_direction_features(data_dict, inference=True)
    datetimes = datetimes[sequence_length:]

    # Create model instance
    model = LSTM(
        input_size=len(feature_columns),
        depth=1,
        width=128,
        dropout=0.2,
        fc_dropout=0.3,
        activation='relu'
    ).to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(f'models/{model._get_model_name()}.pth'))
    print("Loaded trained model")

    # Make predictions
    predictions = predict_cross_current(
        features=features,
        feature_columns=feature_columns,
        model=model,
        device=device,
        norm_params=norm_params,
        sequence_length=sequence_length
    )
    
    print(f"Made predictions for {len(predictions)} time steps")
    print(f"Predictions range: {predictions.min():.3f} to {predictions.max():.3f}")
    print(f"Mean prediction: {predictions.mean():.3f}")
    print(f"Std prediction: {predictions.std():.3f}")
    
    # Plot predictions
    plot_predictions(datetimes, predictions)
    
    return 1

if __name__ == "__main__":
    main()