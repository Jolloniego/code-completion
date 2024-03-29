import torch
import numpy as np
import torch.nn as nn


class BaselineRNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dropout, device):
        super(BaselineRNNModel, self).__init__()
        self.save_name = "BaselineRNNModel.pt"
        self.device = device

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        # Use tanh as non-linearity
        self.rnn = nn.RNN(embedding_dim, 500, batch_first=True)

        self.fc0 = nn.Linear(500, 300)
        self.fc1 = nn.Linear(300, vocab_size)

    def forward(self, input_batch, hidden):
        embeds = self.embeddings(input_batch)
        embeds = self.dropout(embeds)
        outs, hidden = self.rnn(embeds, hidden)
        outs = self.dropout(outs)
        outs = torch.sigmoid(self.fc0(outs[:, -1]))
        outs = self.dropout(outs)
        logits = self.fc1(outs)
        return logits, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, 500).to(self.device)

    @staticmethod
    def detach_hidden(hidden):
        return hidden.detach()

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params
