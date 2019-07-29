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


class AttentionLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dropout, device):
        super(AttentionLSTMModel, self).__init__()
        self.save_name = "AttentionLSTMModel.pt"
        self.device = device

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(embedding_dim, 500, batch_first=True)
        self.attention = Attention(500)

        self.fc0 = nn.Linear(1000, 300)
        self.fc1 = nn.Linear(300, vocab_size)

    def forward(self, input_batch, hidden):
        embeds = self.embeddings(input_batch)
        outs, hidden = self.lstm(embeds, hidden)
        outs = self.dropout(outs)

        attn = self.attention(outs, outs.size(0)).view(outs.size(0), -1)
        outs = torch.cat([attn, outs[:, -1]], dim=1)
        outs = self.dropout(outs)

        outs = torch.sigmoid(self.fc0(outs))
        logits = self.fc1(outs)
        return logits, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, 500).to(self.device), torch.zeros(1, batch_size, 500).to(self.device)

    @staticmethod
    def detach_hidden(hidden):
        return hidden[0].detach(), hidden[1].detach()

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params + self.attention.summary()
