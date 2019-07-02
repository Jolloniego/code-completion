import numpy as np
import torch.nn as nn


class DummyModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(DummyModel, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 128, batch_first=True)
        self.dense = nn.Linear(128, vocab_size)

    def forward(self, code_line):
        embeds = self.embeddings(code_line)
        lstm_out, _ = self.lstm(embeds)
        logits = self.dense(lstm_out)
        return logits

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params
