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
parser.add_argument('--seq_length', type=int, default=10, help='Sequence lengths to use.')
parser.add_argument('--epochs', type=int, default=10, help='Epochs to train for.')

args = parser.parse_args()

word_to_idx = pickle.load(open(args.vocab_path, 'rb'))
# Not needed for now.
# ixd_to_word = {key: word for key, word in enumerate(word_to_idx)}


def prepare_data(data_file_path, batch_size, seq_len):
    start = time.time()
    print("Loading all the data")
    data = du.read_data(data_file_path, args.data_root)
    print("Data loaded in {:.4f} seconds".format(time.time() - start))

    print("Converting text to int")
    start = time.time()
    text_to_int = np.array([[word_to_idx.get(word, du.OOV_IDX) for word in file] for file in data if file != []]).flatten()
    print("Done converting data in {:.4f} seconds".format(time.time() - start))

    print("Creating batches of data")
    start = time.time()
    num_batches = int(len(text_to_int) / (seq_len * batch_size))
    prepared_inputs = text_to_int[:num_batches * batch_size * seq_len]
    prepared_outputs = np.zeros_like(prepared_inputs)
    prepared_outputs[:-1] = prepared_inputs[1:]
    prepared_outputs[-1] = prepared_inputs[0]
    print("Batches completed in {:.4f} seconds".format(time.time() - start))

    return prepared_inputs.reshape((batch_size, -1)), prepared_outputs.reshape((batch_size, -1))


def generate_batches(inputs, outputs, batch_size, seq_len):
    num_batches = np.prod(inputs.shape) // (seq_len * batch_size)
    for i in range(0, num_batches * seq_len, seq_len):
        yield inputs[:, i:i + seq_len], outputs[:, i:i + seq_len]


def train():
    ins, outs = prepare_data(args.train_files, args.batch_size, args.seq_length)

    model = DummyModel(len(word_to_idx), 300).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimiser = optim.Adam(model.parameters(), lr=0.001)

    start = time.time()

    for epoch in range(args.epochs):
        model.train()

        epoch_loss = 0
        for x, y in generate_batches(ins, outs, args.batch_size, args.seq_length):
            model.zero_grad()

            x = torch.LongTensor(x).to(device)
            y = torch.LongTensor(y).to(device)

            # Get the predictions and compute the loss
            preds = model(x)
            loss = criterion(preds.view(-1, 90), y.view(-1))

            # Backprop the loss and update params
            loss.backward()
            optimiser.step()

            epoch_loss += loss.data

        print("Epoch {} | Loss {:.10} | Time taken {:.2f} seconds".format(epoch, epoch_loss, time.time() - start))

    print("Done Training, total time taken: ", time.time() - start)


if __name__ == '__main__':
    device = torch.device(args.cuda if (args.cuda is not None and torch.cuda.is_available()) else 'cpu')
    if args.mode == 'train':
        train()
