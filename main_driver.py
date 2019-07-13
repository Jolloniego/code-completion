import os
import pickle
import argparse

import torch
import numpy as np

import test_driver
import train_driver
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


if __name__ == '__main__':
    device = torch.device(args.cuda if (args.cuda is not None and torch.cuda.is_available()) else 'cpu')
    if args.mode == 'train':
        trained_model = train_driver.train(get_model(args.model), word_to_idx, device, args)
        torch.save(trained_model.state_dict(), os.path.join(args.model_path, trained_model.save_name))
        test_driver.next_token_prediction_test(get_model(args.model), word_to_idx, device, args)

    elif args.mode == 'test':
        test_driver.next_token_prediction_test(get_model(args.model), word_to_idx, device, args)

    else:
        print("Unrecognized mode set. Use train or test only.")
