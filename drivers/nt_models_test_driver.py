import time

import torch
import torch.nn as nn

from datasets.full_line_dataset import NextLineCodeDataset, NextLineCodeDatasetBatcher
from datasets.next_token_dataset import NextTokenCodeDataset, NextTokenCodeDatasetBatcher


def next_token_prediction_test(model, word_to_idx, device, model_path, args):
    # Load the model and set it to eval mode.
    model.load_state_dict(torch.load(model_path))
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

        hidden = model.detach_hidden(hidden)

        # Advance to the next batch
        sample, file_changed = test_dataset_batcher.get_batch()

    print("Next-Token Test Set | Accuracy {:.2f} % | Time taken {:.2f} seconds".format(correct / total * 100, time.time() - start))


def next_line_prediction_test(model, word_to_idx, device, model_path, args):
    # Load the model and set it to eval mode.
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Get the data
    test_dataset = NextLineCodeDataset(args.test_files, args.data_root, args.seq_length, args.prev_lines, word_to_idx)
    test_dataset_batcher = NextLineCodeDatasetBatcher(test_dataset, args.batch_size)

    start = time.time()

    correct = 0
    total = 0
    sample, file_changed = test_dataset_batcher.get_batch()
    while sample is not None:

        for idx, current_input in enumerate(sample[0]):

            previous_tokens = torch.tensor(current_input, device=device).unsqueeze(0)
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
                predictions, hidden = model(predicted_word.unsqueeze(0).to(device), hidden)

            final_output = torch.cat(final_output)
            total += 1
            correct += 1 if torch.equal(final_output, y) else 0

            hidden = model.detach_hidden(hidden)

        # Advance to the next batch
        sample, file_changed = test_dataset_batcher.get_batch()

    print("Next-Line Test Set | Accuracy {:.2f} % | Time taken {:.2f} seconds"
          .format(correct / total * 100, time.time() - start))

