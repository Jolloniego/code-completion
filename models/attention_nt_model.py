import torch
import numpy as np
import torch.nn as nn
from models.attention import Attention


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
        outs = self.dropout(outs)
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
