import time
import torch
import pickle
import argparse
import torch.nn as nn
import torch.optim as optim
from model import DummyModel
from code_dataset import CodeDataset, CodeDatasetBatcher

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


def train():
    dataset = CodeDataset(args.train_files, args.data_root, args.seq_length, word_to_idx)
    dataset_batcher = CodeDatasetBatcher(dataset, args.batch_size)

    model = DummyModel(len(word_to_idx), 300).to(device)
    print("The model has {} trainable parameters.".format(model.summary()))
    criterion = nn.CrossEntropyLoss().to(device)
    optimiser = optim.Adam(model.parameters(), lr=0.001)

    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0

        # Get current training batch
        sample = dataset_batcher.get_batch()
        while sample is not None:
            optimiser.zero_grad()

            x = torch.LongTensor(sample[0]).to(device)
            y = torch.LongTensor(sample[1]).to(device)

            # Get the predictions and compute the loss
            preds = model(x)
            loss = criterion(preds.view(-1, len(word_to_idx)), y.view(-1))

            # Backprop the loss and update params
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item() / len(x)

            # Get the next batch
            sample = dataset_batcher.get_batch()

        print("Epoch {} | Loss {:.10} | Time taken {:.2f} seconds".format(epoch, epoch_loss, time.time() - start))

        # Reset the batcher
        dataset_batcher.reset_batcher()

    print("Done Training, total time taken: ", time.time() - start)


if __name__ == '__main__':
    device = torch.device(args.cuda if (args.cuda is not None and torch.cuda.is_available()) else 'cpu')
    if args.mode == 'train':
        train()
