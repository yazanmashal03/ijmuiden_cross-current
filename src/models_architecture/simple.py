import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    """
    Simplified one-layer LSTM for debugging.
    Input:  (batch, seq_len, input_size)
    Output: (batch, 1)
    """

    def _get_model_name(self):
        return "simple_lstm"
    
    def __init__(self, input_size: int, hidden_size: int = 64):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size

        # Single-layer LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )

        # Simple linear head on last output
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        batch = x.size(0)
        # Initialize hidden and cell states to zeros
        h0 = torch.zeros(1, batch, self.hidden_size, device=x.device)
        c0 = torch.zeros(1, batch, self.hidden_size, device=x.device)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))        # (batch, seq_len, hidden_size)
        last_step = lstm_out[:, -1, :]              # (batch, hidden_size)

        # Linear projection to output scalar
        out = self.fc(last_step)                    # (batch, 1)
        return out
