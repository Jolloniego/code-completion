import torch
import numpy as np
import torch.nn as nn
import utils.data_utils as du
import torch.nn.functional as F


class BaselineEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dropout, device):
        super(BaselineEncoder, self).__init__()
        self.device = device
        self.hidden_size = 500

        self.dropout = nn.Dropout(dropout)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, self.hidden_size)

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
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, output_size)

    def forward(self, x, hidden):
        output = self.embedding(x).view(1, 1, -1)
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
        self.vocab_size = vocab_size
        self.save_name = 'BaselineEncoderDecoder.pt'

    def forward(self, encoder_input, target_tensor, encoder_hidden, criterion):
        loss = 0
        for ei in range(encoder_input.size(0)):
            encoder_out, encoder_hidden = self.encoder(encoder_input[ei].view(1, -1), encoder_hidden)

        decoder_input = torch.tensor([du.PAD_IDX], device=self.device)
        decoder_hidden = encoder_hidden

        # Keep track of decoder outputs
        decoder_outputs = torch.zeros(len(target_tensor), dtype=torch.long, device=self.device)

        # Use teacher forcing for now (use target as next input)
        for di in range(target_tensor.size(0)):
            decoder_out, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_out, target_tensor[di].unsqueeze(0))
            decoder_input = target_tensor[di]
            decoder_outputs[di] = decoder_out.data.topk(1, dim=1)[1].item()

        return loss, encoder_hidden, decoder_outputs

    def train(self, mode=True):
        self.encoder.train(mode)
        self.decoder.train(mode)

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def summary(self):
        encoder_parameters = filter(lambda p: p.requires_grad, self.encoder.parameters())
        params = sum([np.prod(p.size()) for p in encoder_parameters])
        decoder_parameters = filter(lambda p: p.requires_grad, self.decoder.parameters())
        params += sum([np.prod(p.size()) for p in decoder_parameters])
        return params
