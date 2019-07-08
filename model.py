import torch
import numpy as np
import torch.nn as nn


class DummyModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, device):
        super(DummyModel, self).__init__()
        self.save_name = "DummyRnnModel.pt"
        self.hidden_dim = 128
        self.device = device

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, batch_first=True)
        self.dense = nn.Linear(self.hidden_dim, vocab_size)
        self.hidden = self.init_hidden(1)

    def forward(self, input_batch):
        self.hidden = self.init_hidden(len(input_batch))
        embeds = self.embeddings(input_batch)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        logits = self.dense(lstm_out)
        return logits

    def init_hidden(self, batch_size):
        return torch.randn(1, batch_size, self.hidden_dim, device=self.device),\
               torch.randn(1, batch_size, self.hidden_dim, device=self.device)

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params
