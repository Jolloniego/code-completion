import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.data_utils import PAD_IDX


class BaselineEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dropout, device):
        super(BaselineEncoder, self).__init__()
        self.device = device
        self.hidden_size = 500

        self.dropout = nn.Dropout(dropout)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, self.hidden_size, batch_first=True)

    def forward(self, x, hidden):
        out = self.embeddings(x)
        out = self.dropout(out)
        out, hidden = self.gru(out, hidden)
        return out, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class BaselineDecoder(nn.Module):
    def __init__(self, output_size, device):
        super(BaselineDecoder, self).__init__()
        self.hidden_size = 500
        self.device = device

        self.embedding = nn.Embedding(output_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, output_size)

    def forward(self, x, hidden):
        output = self.embedding(x).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class BaselineEncoderDecoderModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dropout, device):
        super(BaselineEncoderDecoderModel, self).__init__()
        self.encoder = BaselineEncoder(vocab_size, embedding_dim, dropout, device)
        self.decoder = BaselineDecoder(vocab_size, device)
        self.device = device
        self.train_mode = True
        self.vocab_size = vocab_size
        self.save_name = 'BaselineEncoderDecoder.pt'

    def forward(self, encoder_input, target_tensor, encoder_hidden):
        _, encoder_hidden = self.encoder(encoder_input.view(1, -1), encoder_hidden)

        decoder_hidden = encoder_hidden

        if self.train_mode:
            decoder_input = torch.ones(len(target_tensor), dtype=torch.long, device=self.device)
            decoder_input[1:] = target_tensor[:-1]
            decoder_logits, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_outputs = None

        else:
            # Keep track of decoder outputs for accuracy calculation
            decoder_outputs = torch.zeros(len(target_tensor), dtype=torch.long, device=self.device)
            decoder_logits = torch.zeros((len(target_tensor), self.vocab_size), device=self.device)

            decoder_input = torch.tensor([PAD_IDX], dtype=torch.long, device=self.device)
            for di in range(target_tensor.size(0)):
                decoder_out, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

                decoder_logits[di] = decoder_out.detach()
                decoder_outputs[di] = decoder_out.data.topk(1, dim=1)[1].item()
                decoder_input = decoder_outputs[di].unsqueeze(0)

        return encoder_hidden, decoder_outputs, decoder_logits

    def train(self, mode=True):
        self.train_mode = True
        self.encoder.train(mode)
        self.decoder.train(mode)

    def eval(self):
        self.train_mode = False
        self.encoder.eval()
        self.decoder.eval()

    def summary(self):
        encoder_parameters = filter(lambda p: p.requires_grad, self.encoder.parameters())
        params = sum([np.prod(p.size()) for p in encoder_parameters])
        decoder_parameters = filter(lambda p: p.requires_grad, self.decoder.parameters())
        params += sum([np.prod(p.size()) for p in decoder_parameters])
        return params
