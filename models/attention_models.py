import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AttentionNet(nn.Module):
    def __init__(self, input_features = 10, output_features = 50, num_classes = 1):
        super(AttentionNet, self).__init__()
        self._input_features = input_features
        self._output_features = output_features  # k
        self._num_classes = num_classes


        self.image1 = nn.Linear(self._input_features, self._output_features)
        self.question1 = nn.Linear(self._input_features, self._output_features)
        self.attention1 = nn.Linear(self._output_features, 1)


        self.image2 = nn.Linear(self._input_features, self._output_features)
        self.question2 = nn.Linear(self._input_features, self._output_features)
        self.attention2 = nn.Linear(self._output_features, 1)


        self.answer_dist = nn.Linear(self._input_features, self._num_classes)


        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, image, question):
        irep_1 = self.image1(image)
        qrep_1 = self.question1(question).unsqueeze(dim=1)
        ha_1 = self.tanh(irep_1 + qrep_1)
        ha_1 = self.dropout(ha_1)
        pi_1 = self.softmax(self.attention1(ha_1))
        u_1 = (pi_1 * image).sum(dim=1) + question

        irep_2 = self.image2(image)
        qrep_2 = self.question2(u_1).unsqueeze(dim=1)
        ha_2 = self.tanh(irep_2 + qrep_2)
        ha_2 = self.dropout(ha_2)
        pi_2 = self.softmax(self.attention2(ha_2))
        u_2 = (pi_2 * image).sum(dim=1) + u_1
        h = self.dropout(u_2)

        w_u = self.answer_dist(h)
        # print('W_u shape:  ',w_u.shape)

        return w_u

class AttentionWithContext(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionWithContext, self).__init__()

        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.contx = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, inp):
        # The first expression in the attention mechanism is simply a linear layer that receives
        # the output of the Word-GRU referred here as 'inp' and h_{it} in the paper
        u = torch.tanh_(self.attn(inp))
        # The second expression is...the same but without bias, wrapped up in a Softmax function
        a = F.softmax(self.contx(u), dim=1)
        # And finally, an element-wise multiplication taking advantage of Pytorch's broadcasting abilities
        s = (a * inp).sum(1)
        # we will also return the normalized importance weights
        return a.permute(0, 2, 1), s


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

        self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.rnn = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.word_attn = AttentionWithContext(hidden_dim * 2)

    def forward(self, X, h_n):
        embed = self.word_embed(X.long())
        h_t, h_n = self.rnn(embed, h_n)
        a, s = self.word_attn(h_t)
        return a, s.unsqueeze(1), h_n


class SentAttnNet(nn.Module):
    def __init__(
            self, word_hidden_dim=32, sent_hidden_dim=32, padding_idx=1
    ):
        super(SentAttnNet, self).__init__()

        self.rnn = nn.GRU(
            word_hidden_dim * 2, sent_hidden_dim, bidirectional=True, batch_first=True
        )
        self.sent_attn = AttentionWithContext(sent_hidden_dim * 2)

    def forward(self, X):
        h_t, h_n = self.rnn(X)
        a, v = self.sent_attn(h_t)
        return a.permute(0, 2, 1), v

class HierAttnNet(nn.Module):
    def __init__(
        self,
        vocab_size,
        maxlen_sent,
        maxlen_doc,
        word_hidden_dim=32,
        sent_hidden_dim=32,
        padding_idx=1,
        embed_dim=50,
        embedding_matrix=None,
        num_class=4,
    ):
        super(HierAttnNet, self).__init__()

        self.word_hidden_dim = word_hidden_dim

        self.wordattnnet = WordAttnNet(
            vocab_size=vocab_size,
            hidden_dim=word_hidden_dim,
            padding_idx=padding_idx,
            embed_dim=embed_dim,
            embedding_matrix=embedding_matrix,
        )

        self.sentattnnet = SentAttnNet(
            word_hidden_dim=word_hidden_dim,
            sent_hidden_dim=sent_hidden_dim,
            padding_idx=padding_idx,
        )

        self.fc = nn.Linear(sent_hidden_dim * 2, num_class)

    def forward(self, X):
        x = X.permute(1, 0, 2)
        word_h_n = nn.init.zeros_(torch.Tensor(2, X.shape[0], self.word_hidden_dim))
        if use_cuda:
            word_h_n = word_h_n.cuda()
        # alpha and s Tensor Lists
        word_a_list, word_s_list = [], []
        for sent in x:
            word_a, word_s, word_h_n = self.wordattnnet(sent, word_h_n)
            word_a_list.append(word_a)
            word_s_list.append(word_s)
        # Importance attention weights per word in sentence
        self.sent_a = torch.cat(word_a_list, 1)
        # Sentences representation
        sent_s = torch.cat(word_s_list, 1)
        # Importance attention weights per sentence in doc and document representation
        self.doc_a, doc_s = self.sentattnnet(sent_s)
        return self.fc(doc_s)