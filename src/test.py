# This is the test script for any model. We test it against new data.
from utils.data_utils import *
from models_architecture.lstm import *
from train import evaluate_model, plot_predictions, analyze_feature_importance

def main():
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    sequence_length = 24

    # Load data
    data_path = 'data/test'
    data_dict = load_data(data_path)

    # check if the data is loaded correctly
    if data_dict is None:
        return
    else:
        print(f"Data loaded")

    # Prepare data
    features, targets, feature_columns, target_column, datetimes = prepare_current_direction_features(data_dict)
    test_dataset = CrossCurrentDataset(features, targets, sequence_length=24)
    datetimes = datetimes[sequence_length:]

    # Load model
    model = LSTM(
    input_size=len(feature_columns),
    depth=1,
    width=64,
    activation='relu'
).to(device)
    
    model.load_state_dict(torch.load(f'models/{model._get_model_name()}.pth'))

    # Test model
    metrics, predictions, actuals = evaluate_model(model, test_dataset, device)

    # Plot predictions
    plot_predictions(actuals, predictions, datetimes, "Cross Current Prediction", mode="evaluation", max_time_points=len(predictions))

    # Analyze feature importance
    importance_scores = analyze_feature_importance(model, feature_columns, test_dataset, metrics, device)

    for metric, value in metrics.items():
        print(f"  {metric}: {value:.6f}")
        print("\nTop 5 Most Important Features:")
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:5]):
            print(f"  {i+1}. {feature}: {importance:.6f}")

if __name__ == "__main__":
    main()