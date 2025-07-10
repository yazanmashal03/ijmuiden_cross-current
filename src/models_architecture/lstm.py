import torch
import torch.nn as nn

class ErfActivation(nn.Module):
    def forward(self, x):
        return torch.special.erf(x)

class LSTM(nn.Module):
    """
    Flexible LSTM architecture with variable layers, depth, and dropout.
    """

    def _get_model_name(self):
        return "lstm"
    
    def __init__(self, input_size, depth, width, output_size=1, 
                 dropout=0.2, fc_dropout=0.3, activation='relu'):
        """
        Initialize the flexible LSTM.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden sizes for each LSTM layer
            output_size: Number of output features (default: 1)
            dropout: Dropout rate between LSTM layers
            fc_dropout: Dropout rate in fully connected layers
            activation: Activation function ('relu', 'tanh', 'leaky_relu')
        """
        super(LSTM, self).__init__()
        
        self.input_size = input_size
        self.depth = depth
        self.width = width
        self.output_size = output_size
        self.dropout = dropout
        self.fc_dropout = fc_dropout

        # Activation function mapping
        activation_functions = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'sigmoid': nn.Sigmoid(),
            'erf': ErfActivation()
        }
        
        self.activation = activation_functions.get(activation, nn.ReLU())
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        
        # First LSTM layer
        self.lstm_layers.append(nn.LSTM(
            input_size=input_size,
            hidden_size=width,
            num_layers=1,
            batch_first=True,
            dropout=0
        ))
        
        # Additional LSTM layers
        for i in range(1, depth):
            self.lstm_layers.append(nn.LSTM(
                input_size=width,
                hidden_size=width,
                num_layers=1,
                batch_first=True,
                dropout=0
            ))
        
        # Dropout layers between LSTM layers
        self.lstm_dropout = nn.Dropout(dropout)
        
        # Fully connected layers with configurable depth
        fc_sizes = [width]
        
        self.fc_layers = nn.ModuleList()
        for i in range(len(fc_sizes) - 1):
            self.fc_layers.append(nn.Linear(fc_sizes[i], fc_sizes[i + 1]))
            if i < len(fc_sizes) - 2:
                self.fc_layers.append(nn.Dropout(fc_dropout))
        
        # Output layer
        self.output_layer = nn.Linear(fc_sizes[-1], output_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Pass through LSTM layers
        lstm_output = x
        for i, lstm_layer in enumerate(self.lstm_layers):
            # LSTM forward pass
            lstm_output, (hidden, cell) = lstm_layer(lstm_output)
            
            # Apply dropout (except after the last LSTM layer)
            if i < len(self.lstm_layers) - 1:
                lstm_output = self.lstm_dropout(lstm_output)
        
        # Take the last output from the final LSTM layer
        last_output = lstm_output[:, -1, :]
        
        # Pass through fully connected layers
        fc_output = last_output
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                fc_output = layer(fc_output)
                fc_output = self.activation(fc_output)
            elif isinstance(layer, nn.Dropout):
                fc_output = layer(fc_output)
        
        # Output layer (no activation for regression)
        output = self.output_layer(fc_output)
        
        return output