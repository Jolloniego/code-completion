import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    # Attention layer code inspired from: https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/4
    # Obtained from https://www.kaggle.com/dannykliu/lstm-with-attention-clr-in-pytorch and modified to fit the problem.
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size

        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)

        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, inputs, batch_size):

        # apply attention layer
        weights = torch.bmm(inputs, self.att_weights.permute(1, 0)  # (1, hidden_size)
                                                    .unsqueeze(0)   # (hidden_size, 1)
                                                    .repeat(batch_size, 1, 1))  # (batch_size, hidden_size, 1)

        attentions = torch.softmax(F.relu(weights.squeeze()), dim=-1)

        # Normalize attention scores (weights)
        sums = attentions.sum(-1).unsqueeze(-1)  # sums per row
        attentions = attentions.div(sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze()

        return representations

    def summary(self):
        return self.hidden_size
