import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.data_utils import PAD_IDX
from models.attention import Attention


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


class AttentionDecoder(nn.Module):
    def __init__(self, output_size, device):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = 128
        self.device = device
        self.dropout = nn.Dropout(0.5)

        self.embedding = nn.Embedding(output_size, self.hidden_size)
        self.attention = Attention(self.hidden_size)
        self.gru = nn.GRU(self.hidden_size * 2, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, output_size)

    def forward(self, x, encoder_outs, hidden):
        output = self.embedding(x)
        output = self.dropout(output)

        batch_size = encoder_outs.size(0)
        attention = self.attention(encoder_outs, batch_size).view(batch_size, -1)

        if self.training:
            attention_output = torch.zeros((batch_size, encoder_outs.size(1), self.hidden_size * 2), device=self.device)
            attention_output[:, :, :self.hidden_size] = output
            attention_output[:, :, self.hidden_size:] = attention.unsqueeze(1).repeat((1, output.size(1), 1))
            output = attention_output

        else:
            output = torch.cat((output, attention), dim=1).unsqueeze(0)

        output, hidden = self.gru(output, hidden)
        output = torch.sigmoid(output)
        output = F.log_softmax(self.out(output), dim=2)

        return output, hidden


class AttentionEncoderDecoderModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, seq_length, dropout, device):
        super(AttentionEncoderDecoderModel, self).__init__()
        # Encode normally, and use attention in the decoder
        self.encoder = BaselineEncoder(vocab_size, embedding_dim, dropout, device)
        self.decoder = AttentionDecoder(vocab_size, device)
        self.device = device
        self.train_mode = True
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.save_name = 'AttentionEncoderDecoder.pt'

    def forward(self, input_batch, targets, encoder_hidden):

        if self.train_mode:
            encoder_outs, encoder_hidden = self.encoder(input_batch, encoder_hidden)

            last_encoder_hidden = encoder_hidden.data
            decoder_hidden = encoder_hidden

            first_input = torch.tensor([PAD_IDX], dtype=torch.long, device=self.device).repeat(
                len(input_batch)).unsqueeze(1)
            decoder_input = torch.cat((first_input, targets), dim=1)[:, :-1]
            decoder_logits, decoder_hidden = self.decoder(decoder_input, encoder_outs, decoder_hidden)

        else:
            with torch.no_grad():
                first_input = torch.tensor([PAD_IDX], dtype=torch.long, device=self.device)

                encoder_outs, encoder_hidden = self.encoder(input_batch, encoder_hidden)

                last_encoder_hidden = encoder_hidden.data
                decoder_hidden = encoder_hidden

                decoder_logits = torch.zeros((len(targets), self.seq_length, self.vocab_size),
                                             dtype=torch.float32).to(self.device)
                for idx in range(len(targets)):
                    decoder_input = first_input
                    current_hidden = decoder_hidden[:, idx].unsqueeze(0)
                    for t_idx in range(len(targets[idx][targets[idx] != 1])):
                        logits, current_hidden = self.decoder(decoder_input,
                                                              encoder_outs[idx].unsqueeze(0),
                                                              current_hidden)
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
