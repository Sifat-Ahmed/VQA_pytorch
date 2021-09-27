import torch
import numpy as np
import torch.nn as nn
from models.attention_models import AttentionWithContext


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim = 100, hidden_dim = 32, num_layers=2, out_features = 10, bidirectional=True):
        super(LSTM, self).__init__()
        self._hidden_dim = hidden_dim
        self._embedding_dim = embedding_dim
        self._embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self._lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self._fc = nn.Linear(hidden_dim * 2, out_features)


    def forward(self, text):
        #lengths = [int(l.item()) for l in torch.count_nonzero(text, dim=1)]
        #print(lengths)
        out = self._embedding(text)
        #out = nn.utils.rnn.pack_padded_sequence(out, lengths, batch_first=True, enforce_sorted=False)

        output, (hidden, cell) = self._lstm(out)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        out = self._fc(hidden)

        return out
