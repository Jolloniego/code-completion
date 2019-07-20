import os
import time

import torch
import torch.nn as nn

from datasets.full_line_dataset import NextLineCodeDataset, NextLineCodeDatasetBatcher
from datasets.next_token_dataset import NextTokenCodeDataset, NextTokenCodeDatasetBatcher


def next_token_prediction_test(model, word_to_idx, device, args):
    # Load the model and set it to eval mode.
    model.load_state_dict(torch.load(os.path.join(args.model_path, model.save_name)))
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

    print("Test Set | Accuracy {:.2f} % | Time taken {:.2f} seconds".format(correct / total * 100, time.time() - start))


def next_line_prediction_test(model, word_to_idx, device, args):
    # Load the model and set it to eval mode.
    model.load_state_dict(torch.load(os.path.join(args.model_path, model.save_name)))
    model.eval()

    # Loss criterion
    criterion = nn.NLLLoss()

    # Get the data
    test_dataset = NextLineCodeDataset(args.test_files, args.data_root, args.seq_length, args.prev_lines, word_to_idx)
    test_dataset_batcher = NextLineCodeDatasetBatcher(test_dataset, args.batch_size)

    start = time.time()

    correct = 0
    total = 0
    total_loss = 0
    sample, file_changed = test_dataset_batcher.get_batch()
    while sample is not None:
        for idx in range(len(sample[0])):

            previous_tokens = torch.tensor(sample[0][idx], device=device).view((1, -1))
            y = torch.tensor(sample[1][idx])

            if file_changed:
                # Feeding one word at a time, so hidden size should be 1.
                hidden = model.init_hidden(1)

            # Feed the input (previous) lines one word at the time
            predictions, hidden = model(previous_tokens, hidden)

            final_output = []
            for _ in range(len(y)):
                predicted_word = torch.argmax(torch.softmax(predictions, dim=1), dim=1).detach().cpu()
                final_output.append(predicted_word)
                predictions, hidden = model(predicted_word.unsqueeze(0), hidden)

            final_output = torch.cat(final_output)
            loss = torch.sum(final_output - y).item() / len(y)

            total += 1
            correct += 1 if loss == 0 else 0
            total_loss += loss

        # Advance to the next batch
        sample, file_changed = test_dataset_batcher.get_batch()

    print("Next-Line Test Set | Accuracy {:.2f} % | Loss {:.2f} | Time taken {:.2f} seconds"
          .format(correct / total * 100, total_loss / total, time.time() - start))

