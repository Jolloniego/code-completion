import argparse
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from code_dataset import CodeDataset, CodeDatasetBatcher
from models.baseline_model import BaselineRNNModel

# Fix random seeds for reproducibility
np.random.seed(2019)
torch.manual_seed(2019)

# Paths
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='data/repos',
                    help='Path root folder containing the cloned repositories.')
parser.add_argument('--model_path', type=str, default='saved_models', help='Path to folder where models will be saved.')
# Files containing paths to data
parser.add_argument('--train_files', type=str, default='data/train.txt',
                    help='Path to file containing the training data split.')
parser.add_argument('--val_files', type=str, default='data/validation.txt',
                    help='Path to file containing the validation data split.')
parser.add_argument('--test_files', type=str, default='data/test.txt',
                    help='Path to file containing the test data split.')
# Run configurations
parser.add_argument('--mode', type=str, default='train', help='train or test')
parser.add_argument('--model', type=int, default=0,
                    help='Model to use. 0 = Baseline model.')
parser.add_argument('--vocab_path', type=str, default='data/vocab.p', help='Path to vocab.p file.')
parser.add_argument('--cuda', type=str,
                    help='Cuda card to use, format: "cuda:int_number". Leave unused to use CPU')
parser.add_argument('--epochs', type=int, default=10, help='Epochs to train for.')
parser.add_argument('--val_epochs', type=int, default=1,
                    help='Number of epochs after which to validate on the validation set')
# Hyperparameters
parser.add_argument('--batch_size', type=int, default=32, help='Batch size to use.')
parser.add_argument('--seq_length', type=int, default=20, help='Sequence lengths to use.')
parser.add_argument('--grad_clip', type=float, default=None, help='Gradient clipping.')
parser.add_argument('--lr', type=float, default=0.001, help='Base Learning Rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Inputs Dropout Rate.')
parser.add_argument('--embedding_dim', type=int, default=300, help='Embedding dimension.')


args = parser.parse_args()

word_to_idx = pickle.load(open(args.vocab_path, 'rb'))
# Not needed for now.
# ixd_to_word = {key: word for key, word in enumerate(word_to_idx)}


def get_model(model_id):
    if model_id == 0:
        return BaselineRNNModel(vocab_size=len(word_to_idx), device=device,
                                embedding_dim=args.embedding_dim, dropout=args.dropout).to(device)
    else:
        raise ValueError("Model not known. Use 0 for BaselineRNNModel.")


def train(model):
    # Get training and validation data
    train_dataset = CodeDataset(args.train_files, args.data_root, args.seq_length, word_to_idx)
    val_dataset = CodeDataset(args.val_files, args.data_root, args.seq_length, word_to_idx)
    train_dataset_batcher = CodeDatasetBatcher(train_dataset, args.batch_size)

    # Create the model, optimizer and criterion to use
    print("The model has {} trainable parameters.".format(model.summary()))
    optimiser = optim.Adam(model.parameters(), lr=args.lr)

    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()

        epoch_loss = 0
        correct = 0
        total = 0

        epoch_start = time.time()
        # Get current training batch
        sample, file_changed = train_dataset_batcher.get_batch()
        while sample is not None:
            optimiser.zero_grad()

            x = torch.tensor(sample[0], device=device)
            y = torch.tensor(sample[1], device=device)

            if file_changed:
                hidden = model.init_hidden(len(x))

            # Get the predictions and compute the loss
            preds, hidden = model(x, hidden)
            loss = criterion(preds, y)

            # Track accuracy as well
            total += len(x)
            preds = torch.argmax(nn.functional.softmax(preds, dim=1), dim=1).detach()
            correct += (preds == y).sum().item()

            # Backprop the loss and update params, use gradient clipping if specified
            loss.backward(retain_graph=True)
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimiser.step()

            epoch_loss += loss.item() / len(x)

            # Get the next batch
            sample, file_changed = train_dataset_batcher.get_batch()

        print("Epoch {} | Loss {:.10} | Accuracy {:.2f}% | Time taken {:.2f} seconds"
              .format(epoch, epoch_loss, (correct / total * 100), time.time() - epoch_start))

        # Validate if we need to
        if epoch % args.val_epochs == 0:
            model.eval()
            validate(model, val_dataset, time.time())

        # Reset the batcher
        train_dataset_batcher.reset_batcher()

    print("Done Training, total time taken: ", time.time() - train_start)
    return model


def validate(model, val_dataset, start_time):
    val_dataset_batcher = CodeDatasetBatcher(val_dataset, args.batch_size)

    validation_loss = 0
    total = 0
    correct = 0

    sample, file_changed = val_dataset_batcher.get_batch()
    while sample is not None:
        x = torch.tensor(sample[0], device=device)
        y = torch.tensor(sample[1], device=device)

        if file_changed:
            hidden = model.init_hidden(len(x))

        preds, hidden = model(x, hidden)
        loss = criterion(preds, y).item()
        validation_loss += loss / len(x)

        # Track accuracy
        total += len(x)
        preds = torch.argmax(nn.functional.softmax(preds, dim=1), dim=1).detach()
        correct += (preds == y).sum().item()

        # Advance to the next batch
        sample, file_changed = val_dataset_batcher.get_batch()

    print("Validation epoch | Loss {:.10} | Accuracy {:.2f}% | Time taken {:.2f} seconds"
          .format(validation_loss, (correct / total * 100), time.time() - start_time))


def next_token_prediction_test(model):
    # Load the model and set it to eval mode.
    model.load_state_dict(torch.load(os.path.join(args.model_path, model.save_name)))
    model.eval()

    # Get the data
    test_dataset = CodeDataset(args.test_files, args.data_root, args.seq_length, word_to_idx)
    test_dataset_batcher = CodeDatasetBatcher(test_dataset, args.batch_size)

    start = time.time()

    correct = 0
    total = 0
    sample, file_changed = test_dataset_batcher.get_batch()
    while sample is not None:
        x = torch.tensor(sample[0], device=device)
        y = torch.tensor(sample[1])

        if file_changed:
            hidden = model.init_hidden(len(x))

        preds, hidden = model(x, hidden)
        preds = torch.argmax(nn.functional.softmax(preds, dim=1), dim=1).detach().cpu()

        correct += (preds == y).sum().item()
        total += len(x)
        # Advance to the next batch
        sample, file_changed = test_dataset_batcher.get_batch()

    print("Test Set | Accuracy {:.2f} % | Time taken {:.2f} seconds".format(correct / total * 100, time.time() - start))


if __name__ == '__main__':
    device = torch.device(args.cuda if (args.cuda is not None and torch.cuda.is_available()) else 'cpu')
    criterion = nn.CrossEntropyLoss().to(device)
    if args.mode == 'train':
        trained_model = train(get_model(args.model))
        torch.save(trained_model.state_dict(), os.path.join(args.model_path, trained_model.save_name))
        next_token_prediction_test(get_model(args.model))

    elif args.mode == 'test':
        next_token_prediction_test(get_model(args.model))

    else:
        print("Unrecognized mode set. Use train or test only.")
