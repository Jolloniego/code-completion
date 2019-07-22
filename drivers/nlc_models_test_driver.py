import os
import time

import torch
import torch.nn as nn

from datasets.full_line_dataset import NextLineCodeDataset, NextLineCodeDatasetBatcher
from datasets.next_token_dataset import NextTokenCodeDataset, NextTokenCodeDatasetBatcher


def next_token_prediction_test(model, word_to_idx, device, model_name, args):
    # Load the model and set it to eval mode.
    model.load_state_dict(torch.load(os.path.join(args.model_path, model_name)))
    model.eval()

    # Get the data
    test_dataset = NextTokenCodeDataset(args.test_files, args.data_root, args.seq_length, word_to_idx)
    test_dataset_batcher = NextTokenCodeDatasetBatcher(test_dataset, args.batch_size)

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

    print("Next-Token Test Set | Accuracy {:.2f} % | Time taken {:.2f} seconds".format(correct / total * 100, time.time() - start))


def next_line_prediction_test(model, word_to_idx, device, model_name, args):
    # Load the model and set it to eval mode.
    model.load_state_dict(torch.load(os.path.join(args.model_path, model_name)))
    model.eval()

    # Get the data
    test_dataset = NextLineCodeDataset(args.test_files, args.data_root, args.seq_length, args.prev_lines, word_to_idx)
    test_dataset_batcher = NextLineCodeDatasetBatcher(test_dataset, args.batch_size)

    start = time.time()

    correct = 0
    total = 0
    total_loss = 0
    sample, file_changed = test_dataset_batcher.get_batch()
    while sample is not None:

        if file_changed:
            hidden = model.init_hidden(args.seq_length)

        for idx, current_input in enumerate(sample[0]):

            x = torch.tensor(current_input, device=device)
            y = torch.tensor(sample[1][idx], device=device)

            preds, hidden = model(x, hidden)

            # Track accuracy
            total += 1
            preds = torch.argmax(nn.functional.softmax(preds, dim=1), dim=1).detach()
            correct += 1 if torch.equal(preds, y) else 0

        # Advance to the next batch
        sample, file_changed = test_dataset_batcher.get_batch()

    print("Next-Line Test Set | Accuracy {:.2f} % | Time taken {:.2f} seconds"
          .format(correct / total * 100, time.time() - start))

