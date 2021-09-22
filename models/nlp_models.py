import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, embedding_dim = 300, hidden_dim = 128, num_layers=2, out_features = 100, bidirectional=True):
        super(LSTM, self).__init__()
        self._hidden_dim = hidden_dim
        self._embedding_dim = embedding_dim
        self._lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self._fc = nn.Linear(hidden_dim * 2, out_features)


    def forward(self, text):
        output, (hidden, cell) = self._lstm(text)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        out = self._fc(hidden)

        return out