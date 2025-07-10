from utils.data_utils import *
from models_architecture.lstm import *
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os

def train_model(train_dataset, val_dataset, model, model_config, device):
    """
    Train the neural network model.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        model_config: Model configuration parameters
        
    Returns:
        Tuple of (model, training_history)
    """
    
    print("Training neural network model for current direction prediction...")
    print(f"Model config: {model_config}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=model_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=model_config['batch_size'], shuffle=False)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=model_config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.10)
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(model_config['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'models/{model._get_model_name()}.pth')
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{model_config["epochs"]}], '
                  f'Train Loss: {train_loss:.6f}, '
                  f'Val Loss: {val_loss:.6f}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Early stopping
        if patience_counter >= model_config['patience']:
            print(f'Early stopping at epoch {epoch+1}')
            break

        # Stop if the learning rate is too low
        if optimizer.param_groups[0]['lr'] < 1e-6:
            print(f'Learning rate too low at epoch {epoch+1}')
            break
    
    # Load best model
    model.load_state_dict(torch.load(f'models/{model._get_model_name()}.pth'))
    
    return model, {'train_losses': train_losses, 'val_losses': val_losses}

def plot_training_history(history):
    """Plot training and validation loss."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Training Loss', color='blue')
    plt.plot(history['val_losses'], label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(history['train_losses'], label='Training Loss', color='blue')
    plt.semilogy(history['val_losses'], label='Validation Loss', color='red')
    plt.title('Training and Validation Loss (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, test_dataset, device):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        scaler: Optional scaler for denormalization
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(batch_y.numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    print("Current Direction Model Performance:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    return metrics, predictions, actuals

def plot_predictions(actuals, predictions, title="Current Direction Predictions vs Actual"):
    """Plot predictions against actual values."""
    plt.figure(figsize=(15, 10))
    
    # Scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
    plt.xlabel('Actual Cross Current')
    plt.ylabel('Predicted Cross Current')
    plt.title(f'{title} - Scatter Plot')
    plt.grid(True)
    
    # Time series plot (first 200 points)
    plt.subplot(2, 2, 2)
    n_points = min(200, len(actuals))
    plt.plot(actuals[:n_points], label='Actual', alpha=0.7)
    plt.plot(predictions[:n_points], label='Predicted', alpha=0.7)
    plt.xlabel('Time Steps')
    plt.ylabel('Cross Current')
    plt.title(f'{title} - Time Series (First {n_points} points)')
    plt.legend()
    plt.grid(True)
    
    # Residuals plot
    plt.subplot(2, 2, 3)
    residuals = actuals - predictions
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Cross Current')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.grid(True)
    
    # Residuals histogram
    plt.subplot(2, 2, 4)
    plt.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def analyze_feature_importance(model, feature_columns, test_dataset, device):
    """Analyze feature importance using permutation importance."""
    
    # Get baseline performance
    baseline_metrics, _, _ = evaluate_model(model, test_dataset, device)
    baseline_mse = baseline_metrics['MSE']
    
    importance_scores = {}
    
    # Test each feature
    for i, feature_name in enumerate(feature_columns):
        print(f"Testing feature: {feature_name}")
        
        # Create modified test dataset with shuffled feature
        modified_data = test_dataset.data.clone()
        modified_data[:, i] = torch.randn_like(modified_data[:, i])
        
        modified_dataset = CrossCurrentDataset(
            modified_data.numpy(), 
            test_dataset.targets.numpy(), 
            test_dataset.sequence_length
        )
        
        # Evaluate with modified feature
        modified_metrics, _, _ = evaluate_model(model, modified_dataset, device)
        modified_mse = modified_metrics['MSE']
        
        # Importance is the increase in MSE
        importance = modified_mse - baseline_mse
        importance_scores[feature_name] = importance
    
    # Sort features by importance
    sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    feature_names = [f[0] for f in sorted_features]
    importance_values = [f[1] for f in sorted_features]
    
    plt.barh(range(len(feature_names)), importance_values)
    plt.yticks(range(len(feature_names)), feature_names)
    plt.xlabel('Feature Importance (MSE Increase)')
    plt.title('Feature Importance for Current Direction Prediction')
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.show()
    
    return importance_scores


def main():
    """Main function to train the model."""
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    data_path = 'data/processed'
    data_dict = load_data(data_path)

    # Prepare data
    features, targets, feature_columns, target_column = prepare_current_direction_features(data_dict)

    # Create datasets
    sequence_length = 24
    train_dataset, val_dataset, test_dataset = create_datasets(features, targets, sequence_length=sequence_length)

    # Initialize model
    input_size = train_dataset.data.shape[1]
    model_config = {
        'depth': 1,
        'width': 128,
        'dropout': 0.2,
        'fc_dropout': 0.3,
        'activation': 'relu',
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.01,
        'patience': np.inf,
    }

    model = LSTM(
        input_size=input_size,
        depth=model_config['depth'],
        width=model_config['width'],
        dropout=model_config['dropout'],
        fc_dropout=model_config['fc_dropout'],
        activation=model_config['activation']
    ).to(device)
    
    # Train model
    model, history = train_model(train_dataset, val_dataset, model,  model_config, device)

    # Plot training history
    plot_training_history(history)

    # Evaluate model
    metrics, predictions, actuals = evaluate_model(model, test_dataset, device)

    # Plot predictions
    plot_predictions(actuals, predictions, "Cross Current Prediction")

    # Analyze feature importance
    importance_scores = analyze_feature_importance(model, feature_columns, test_dataset, device)

    for metric, value in metrics.items():
        print(f"  {metric}: {value:.6f}")
        print("\nTop 5 Most Important Features:")
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:5]):
            print(f"  {i+1}. {feature}: {importance:.6f}")
        print("\nModel Architecture:")
        print(f"  LSTM Layers: {model_config['width']}")
        print(f"  Hidden Size: {model_config['depth']}")
        print(f"  Dropout: {model_config['dropout']}")
        print(f"  Learning Rate: {model_config['learning_rate']}")
        print("="*60)

if __name__ == "__main__":
    main()