import math
import time
import torch
import pickle
import argparse
import itertools
import numpy as np
import torch.nn as nn
import data_utils as du
import torch.optim as optim
from model import DummyModel

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='data/repos',
                    help='Path root folder containing the cloned repositories.')
parser.add_argument('--vocab_path', type=str, default='data/vocab.p', help='Path to vocab.p file.')
# Files containing paths to data
parser.add_argument('--train_files', type=str, default='data/train.txt',
                    help='Path to file containing the training data split.')
parser.add_argument('--val_files', type=str, default='data/validation.txt',
                    help='Path to file containing the validation data split.')
parser.add_argument('--test_files', type=str, default='data/test.txt',
                    help='Path to file containing the test data split.')
# Run configurations
parser.add_argument('--mode', type=str, default='train', help='train or test')
parser.add_argument('--cuda', type=str,
                    help='Cuda card to use, format: "cuda:int_number". Leave unused to use CPU')
# Hyperparameters
parser.add_argument('--batch_size', type=int, default=32, help='Batch size to use.')
parser.add_argument('--seq_length', type=int, default=20, help='Sequence lengths to use.')
parser.add_argument('--epochs', type=int, default=10, help='Epochs to train for.')

args = parser.parse_args()

word_to_idx = pickle.load(open(args.vocab_path, 'rb'))
# Not needed for now.
# ixd_to_word = {key: word for key, word in enumerate(word_to_idx)}


def vecotrize_and_pad_data(data):
    result = []
    data = np.array(list(itertools.chain(*data)))
    newlines = np.where(data == '\n')[0]

    start_idx = 0
    for newline_idx in newlines:
        current_line = [word_to_idx.get(word, du.OOV_IDX) for word in data[start_idx:newline_idx][:args.seq_length]]
        current_line = np.pad(current_line, (0, args.seq_length - len(current_line)), mode='constant', constant_values=du.PAD_IDX)

        result.append(current_line)
        start_idx = newline_idx + 1

    return np.array(result)


def prepare_data(data_file_path):
    start = time.time()
    print("Loading all the data")
    data = du.read_data(data_file_path, args.data_root)
    data = vecotrize_and_pad_data(data)
    print("Data loaded and padded/trimmed in {:.4f} seconds".format(time.time() - start))

    prepared_outputs = np.zeros_like(data)
    prepared_outputs[:-1] = data[1:]
    prepared_outputs[-1] = data[0]

    return data, prepared_outputs


def generate_batches(inputs, outputs):
    num_batches = math.ceil(len(inputs) / args.batch_size)
    for i in range(0, num_batches):
        start_idx = i * args.batch_size
        end_idx = args.batch_size + start_idx
        yield inputs[start_idx: end_idx], outputs[start_idx: end_idx]


def train():
    ins, outs = prepare_data(args.train_files)

    model = DummyModel(len(word_to_idx), 300).to(device)
    print("The model has {} trainable parameters.".format(model.summary()))
    criterion = nn.CrossEntropyLoss().to(device)
    optimiser = optim.Adam(model.parameters(), lr=0.001)

    start = time.time()

    for epoch in range(args.epochs):
        model.train()

        epoch_loss = 0
        for x, y in generate_batches(ins, outs):
            optimiser.zero_grad()

            x = torch.LongTensor(x).to(device)
            y = torch.LongTensor(y).to(device)

            # Get the predictions and compute the loss
            preds = model(x)
            loss = criterion(preds.view(-1, len(word_to_idx)), y.view(-1))

            # Backprop the loss and update params
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()

        print("Epoch {} | Loss {:.10} | Time taken {:.2f} seconds".format(epoch, epoch_loss, time.time() - start))

    print("Done Training, total time taken: ", time.time() - start)


if __name__ == '__main__':
    device = torch.device(args.cuda if (args.cuda is not None and torch.cuda.is_available()) else 'cpu')
    if args.mode == 'train':
        train()
