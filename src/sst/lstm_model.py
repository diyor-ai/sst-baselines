"""Small LSTM: one layer, last hidden state -> linear layer -> 3 outputs (the anomalies at t+1, t+7, t+30)."""
import torch
from torch import nn


class SSTLSTM(nn.Module):
    def __init__(self, hidden=32, n_in=3, n_out=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_in, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x):                    # x: (batch, 30, 3)
        out, _ = self.lstm(x)                # out: (batch, 30, hidden) = hidden state after every input day
        return self.head(out[:, -1, :])      # use only the state after the LAST day (t) -> (batch, 3)
