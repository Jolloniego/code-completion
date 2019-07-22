import argparse
import os
import pickle

import numpy as np
import torch

from models.lstm_model import LSTMModel
from models.baseline_model import BaselineRNNModel
from drivers import nt_models_test_driver, nt_train_driver, nlc_train_driver

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
parser.add_argument('--mode', type=int, default=0,
                    help='0 - Train model on next-token prediction data.\n'
                         '1 - Train model on next-line prediction data.\n'
                         '2 - Test models trained on next-token tasks.\n'
                         '3 - Test models trained on next-line prediction.\n')
parser.add_argument('--model', type=int, default=0,
                    help='Model to use. 0 = Baseline model. 1 = Basic LSTM model.')
parser.add_argument('--vocab_path', type=str, default='data/vocab.p', help='Path to vocab.p file.')
parser.add_argument('--cuda', type=str,
                    help='Cuda card to use, format: "cuda:int_number". Leave unused to use CPU')
parser.add_argument('--epochs', type=int, default=10, help='Epochs to train for.')
parser.add_argument('--val_epochs', type=int, default=1,
                    help='Number of epochs after which to validate on the validation set')
# Hyperparameters
parser.add_argument('--batch_size', type=int, default=32, help='Batch size to use.')
parser.add_argument('--seq_length', type=int, default=20, help='Sequence lengths to use.')
parser.add_argument('--prev_lines', type=int, default=32,
                    help='Number of previous lines to use for next line prediction.')
parser.add_argument('--grad_clip', type=float, default=None, help='Gradient clipping.')
parser.add_argument('--lr', type=float, default=0.001, help='Base Learning Rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Inputs Dropout Rate.')
parser.add_argument('--embedding_dim', type=int, default=300, help='Embedding dimension.')


args = parser.parse_args()

word_to_idx = pickle.load(open(args.vocab_path, 'rb'))
device = torch.device(args.cuda if (args.cuda is not None and torch.cuda.is_available()) else 'cpu')

# Not needed for now.
# idx_to_word = {key: word for key, word in enumerate(word_to_idx)}


def get_model(model_id):
    """
    Returns the object associated to the model selected in the args.
    """
    if model_id == 0:
        return BaselineRNNModel(vocab_size=len(word_to_idx), device=device,
                                embedding_dim=args.embedding_dim, dropout=args.dropout).to(device)
    elif model_id == 1:
        return LSTMModel(vocab_size=len(word_to_idx), device=device,
                         embedding_dim=args.embedding_dim, dropout=args.dropout).to(device)
    else:
        raise ValueError("Model not known. Use 0 for BaselineRNNModel. 1 for BasicLSTMModel.")


def train_model_next_token():
    """
    Trains the selected model (from args) for the next token prediction task on the next-token dataset.
    After training, saves the model to disk and runs the corresponding test suite.
    """
    trained_model = nt_train_driver.train(get_model(args.model), word_to_idx, device, args)
    model_save_name = mode_names[args.mode] + '_' + trained_model.save_name
    torch.save(trained_model.state_dict(), os.path.join(args.model_path, model_save_name))
    next_token_models_tests()


def train_model_next_line():
    """
    Trains the selected model for the next line of code prediction task on the next line dataset.
    After training saves the model on disk and runs the corresponding tests.
    """
    trained_model = nlc_train_driver.train(get_model(args.model), word_to_idx, device, args)
    model_save_name = mode_names[args.mode] + '_' + trained_model.save_name
    torch.save(trained_model.state_dict(), os.path.join(args.model_path, model_save_name))
    next_line_models_tests()


def next_token_models_tests():
    """
    Tests a model trained for the next token prediction task on both the next token prediction and the
    next line of code prediction.
    """
    model = get_model(args.model)
    model_save_name = mode_names[args.mode] + '_' + model.save_name
    nt_models_test_driver.next_token_prediction_test(model, word_to_idx, device, model_save_name, args)
    nt_models_test_driver.next_line_prediction_test(model, word_to_idx, device, model_save_name, args)


def next_line_models_tests():
    return None


# Operates as a switch for the different modes.
#  Functions without () so they are not executed when declared.
mode_functions = {
    0: train_model_next_token,
    1: train_model_next_line,
    2: next_token_models_tests,
    3: next_line_models_tests
}
mode_names = {
    0: 'NT',
    1: 'NLC',
    2: 'NT',
    3: 'NLC'
}

if __name__ == '__main__':
    try:
        mode_functions[args.mode]()
    except KeyError:
        print('Unrecognised mode. Available modes are:\n'
              '0 - Train model on next-token prediction data.\n'
              '1 - Train model on next-line prediction data.\n'
              '2 - Test models trained on next-token tasks.\n'
              '3 - Test models trained on next-line prediction.\n')
