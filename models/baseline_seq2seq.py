import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.data_utils import PAD_IDX


class BaselineEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dropout, device):
        super(BaselineEncoder, self).__init__()
        self.device = device
        self.hidden_size = 128

        self.dropout = nn.Dropout(dropout)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, self.hidden_size, batch_first=True)

    def forward(self, x, hidden):
        out = self.embeddings(x)
        out = self.dropout(out)
        out, hidden = self.gru(out, hidden)
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)


class BaselineDecoder(nn.Module):
    def __init__(self, output_size, device):
        super(BaselineDecoder, self).__init__()
        self.hidden_size = 128
        self.device = device

        self.embedding = nn.Embedding(output_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, output_size)

    def forward(self, x, hidden):
        output = self.embedding(x)
        output = F.relu(output)

        if self.training:
            output, hidden = self.gru(output, hidden)
        else:
            output, hidden = self.gru(output.view(1, 1, -1), hidden)

        output = F.log_softmax(self.out(output), dim=2)

        return output, hidden


class BaselineEncoderDecoderModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, seq_length, dropout, device):
        super(BaselineEncoderDecoderModel, self).__init__()
        self.encoder = BaselineEncoder(vocab_size, embedding_dim, dropout, device)
        self.decoder = BaselineDecoder(vocab_size, device)
        self.device = device
        self.train_mode = True
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.save_name = 'BaselineEncoderDecoder.pt'

    def forward(self, input_batch, targets, encoder_hidden):

        if self.train_mode:
            _, encoder_hidden = self.encoder(input_batch, encoder_hidden)

            last_encoder_hidden = encoder_hidden.data
            decoder_hidden = encoder_hidden

            first_input = torch.tensor([PAD_IDX], dtype=torch.long, device=self.device).repeat(
                len(input_batch)).unsqueeze(1)
            decoder_input = torch.cat((first_input, targets), dim=1)[:, :-1]
            decoder_logits, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

        else:
            with torch.no_grad():
                first_input = torch.tensor([PAD_IDX], dtype=torch.long, device=self.device)

                _, encoder_hidden = self.encoder(input_batch, encoder_hidden)

                last_encoder_hidden = encoder_hidden.data
                decoder_hidden = encoder_hidden

                decoder_logits = torch.zeros((len(targets), self.seq_length, self.vocab_size), dtype=torch.float32)
                for idx in range(len(targets)):
                    decoder_input = first_input
                    current_hidden = decoder_hidden[:, idx].unsqueeze(0)
                    for t_idx in range(len(targets[idx])):
                        logits, current_hidden = self.decoder(decoder_input, current_hidden)
                        decoder_logits[idx, t_idx] = logits.data
                        # Feed its previous output as next input
                        decoder_input = logits.data.topk(1)[1].view(1)

        return last_encoder_hidden, decoder_logits

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
