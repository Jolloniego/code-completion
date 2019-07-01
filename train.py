import os
import time
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
import data_utils as du
import torch.optim as optim
from model import DummyModel

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='data/repos',
                    help='Path root folder containing the cloned repositories.')
parser.add_argument('--files_split', type=str, default='data',
                    help='Path to the directory containing the train, test and valid.txt files')
parser.add_argument('--cuda', type=str,
                    help='Cuda card to use, format: "cuda:int_number". Leave unused to use CPU')
parser.add_argument('--mode', type=str, default='train', help='train or test')

args = parser.parse_args()

BATCH_SIZE = 32
SEQ_LEN = 10
NUM_EPOCHS = 30

word_to_idx = pickle.load(open('data/vocab.p', 'rb'))

# Not needed for now.
# ixd_to_word = {key: word for key, word in enumerate(word_to_idx)}


def prepare_data(data_file_path, batch_size, seq_len):
    data = du.read_data(data_file_path, args.data_root)

    text_to_int = np.array([[word_to_idx[word] for word in file] for file in data if file != []]).flatten()

    num_batches = int(len(text_to_int) / (seq_len * batch_size))
    prepared_inputs = text_to_int[:num_batches * batch_size * seq_len]
    prepared_outputs = np.zeros_like(prepared_inputs)
    prepared_outputs[:-1] = prepared_inputs[1:]
    prepared_outputs[-1] = prepared_inputs[0]

    return prepared_inputs.reshape((batch_size, -1)), prepared_outputs.reshape((batch_size, -1))


def generate_batches(inputs, outputs, batch_size, seq_len):
    num_batches = np.prod(inputs.shape) // (seq_len * batch_size)
    for i in range(0, num_batches * seq_len, seq_len):
        yield inputs[:, i:i + seq_len], outputs[:, i:i + seq_len]


def train():
    ins, outs = prepare_data(os.path.join(args.files_split, 'train.txt'), BATCH_SIZE, SEQ_LEN)

    model = DummyModel(len(word_to_idx), 300)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=0.001)

    start = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()

        epoch_loss = 0
        for x, y in generate_batches(ins, outs, BATCH_SIZE, SEQ_LEN):
            model.zero_grad()

            x = torch.LongTensor(x)
            y = torch.LongTensor(y)

            # Get the predictions and compute the loss
            preds = model(x)
            loss = criterion(preds.view(-1, 90), y.view(-1))

            # Backprop the loss and update params
            loss.backward()
            optimiser.step()

            epoch_loss += loss.data

        print("Epoch {} | Loss {} | Time taken {:.2f} seconds".format(epoch, epoch_loss, time.time() - start))

    print("Done Training, total time taken: ", time.time() - start)


if __name__ == '__main__':
    device = torch.device(args.cuda if (args.cuda is not None and torch.cuda.is_available()) else 'cpu')
    if args.mode == 'train':
        train()
