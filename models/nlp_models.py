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

class WordAttnNet(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim=32,
        padding_idx=1,
        embed_dim=50,
        embedding_matrix=None,
    ):
        super(WordAttnNet, self).__init__()

        if isinstance(embedding_matrix, np.ndarray):
            self.word_embed = nn.Embedding(
                vocab_size, embedding_matrix.shape[1], padding_idx=padding_idx
            )
            self.word_embed.weight = nn.Parameter(torch.Tensor(embedding_matrix))
            embed_dim = embedding_matrix.shape[1]
        else:
            self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        self.rnn = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True)

        self.word_attn = AttentionWithContext(hidden_dim * 2)

    def forward(self, X, h_n):
        embed = self.word_embed(X.long())
        h_t, h_n = self.rnn(embed, h_n)
        a, s = self.word_attn(h_t)
        return a, s.unsqueeze(1), h_n