import torch
import numpy as np
import torch.nn as nn


class BaselineRNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, device):
        super(BaselineRNNModel, self).__init__()
        self.save_name = "BaselineRNNModel.pt"
        self.device = device

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(0.5)
        # Use tanh as non-linearity
        self.rnn = nn.RNN(embedding_dim, 500, batch_first=True)

        self.fc0 = nn.Linear(500, 300)
        self.fc1 = nn.Linear(300, vocab_size)

    def forward(self, input_batch):
        embeds = self.embeddings(input_batch)
        embeds = self.dropout(embeds)
        outs, _ = self.rnn(embeds)
        outs = torch.sigmoid(self.fc0(outs[:, -1]))
        logits = self.fc1(outs)
        return logits

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params