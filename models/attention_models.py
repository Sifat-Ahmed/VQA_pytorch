import torch
import torch.nn as nn


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