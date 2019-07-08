import time
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from model import DummyModel
from code_dataset import CodeDataset, CodeDatasetBatcher

# Fix random seeds for reproducibility
np.random.seed(2019)
torch.manual_seed(2019)

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
parser.add_argument('--val_epochs', type=int, default=1,
                    help='Number of epochs after which to validate on the validation set')
# Hyperparameters
parser.add_argument('--batch_size', type=int, default=32, help='Batch size to use.')
parser.add_argument('--seq_length', type=int, default=20, help='Sequence lengths to use.')
parser.add_argument('--epochs', type=int, default=10, help='Epochs to train for.')
parser.add_argument('--grad_clip', type=int, default=None, help='Gradient clipping.')

args = parser.parse_args()

word_to_idx = pickle.load(open(args.vocab_path, 'rb'))
# Not needed for now.
# ixd_to_word = {key: word for key, word in enumerate(word_to_idx)}


def train():
    # Get training and validation data
    train_dataset = CodeDataset(args.train_files, args.data_root, args.seq_length, word_to_idx)
    val_dataset = CodeDataset(args.val_files, args.data_root, args.seq_length, word_to_idx)
    train_dataset_batcher = CodeDatasetBatcher(train_dataset, args.batch_size)

    # Create the model, optimizer and criterion to use
    model = DummyModel(len(word_to_idx), 300, device).to(device)
    print("The model has {} trainable parameters.".format(model.summary()))
    criterion = nn.CrossEntropyLoss().to(device)
    optimiser = optim.Adam(model.parameters(), lr=0.001)

    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0

        # Get current training batch
        sample = train_dataset_batcher.get_batch()
        while sample is not None:

            optimiser.zero_grad()

            x = torch.tensor(sample[0], device=device)
            y = torch.tensor(sample[1], device=device)

            # Get the predictions and compute the loss
            preds = model(x)
            loss = criterion(preds.view(-1, len(word_to_idx)), y.view(-1))

            # Backprop the loss and update params, use gradient clipping if specified
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimiser.step()

            epoch_loss += loss.item() / len(x)

            # Get the next batch
            sample = train_dataset_batcher.get_batch()

        print("Epoch {} | Loss {:.10} | Time taken {:.2f} seconds".format(epoch, epoch_loss, time.time() - start))

        # Validate if we need to
        if epoch % args.val_epochs == 0:
            validate(model, criterion, val_dataset, time.time())

        # Reset the batcher
        train_dataset_batcher.reset_batcher()

    print("Done Training, total time taken: ", time.time() - start)


def validate(model, criterion, val_dataset, start_time):
    val_dataset_batcher = CodeDatasetBatcher(val_dataset, args.batch_size)

    validation_loss = 0
    sample = val_dataset_batcher.get_batch()
    while sample is not None:
        x = torch.tensor(sample[0], device=device)
        y = torch.tensor(sample[1], device=device)

        preds = model(x)
        loss = criterion(preds.view(-1, len(word_to_idx)), y.view(-1)).item()
        validation_loss += loss / len(x)

        # Advance to the next batch
        sample = val_dataset_batcher.get_batch()

    print("Validation epoch | Loss {:.10} | Time taken {:.2f} seconds".format(validation_loss, time.time() - start_time))


if __name__ == '__main__':
    device = torch.device(args.cuda if (args.cuda is not None and torch.cuda.is_available()) else 'cpu')
    if args.mode == 'train':
        train()
