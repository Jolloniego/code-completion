import time

import torch
import torch.nn as nn

from utils.data_utils import PAD_IDX
from datasets.full_line_dataset import NextLineCodeDataset, NextLineCodeDatasetBatcher
from datasets.next_token_dataset import NextTokenCodeDataset, NextTokenCodeDatasetBatcher


def next_token_prediction_test(model, word_to_idx, device, args):
    model.eval()

    # Get the data
    test_dataset = NextTokenCodeDataset(args.test_files, args.data_root, args.seq_length, word_to_idx)
    test_dataset_batcher = NextTokenCodeDatasetBatcher(test_dataset, args.batch_size)

    start = time.time()

    correct = 0
    top_3_correct = 0
    total = 0
    sample, file_changed = test_dataset_batcher.get_batch()
    while sample is not None:
        x = torch.tensor(sample[0], device=device)
        y = torch.tensor(sample[1])

        if file_changed:
            hidden = model.init_hidden(len(x))

        preds, hidden = model(x, hidden)
        top3_preds = nn.functional.softmax(preds, dim=1).topk(3)[1].cpu()
        preds = torch.argmax(nn.functional.softmax(preds, dim=1), dim=1).detach().cpu()

        correct += (preds == y).sum().item()

        for idx, target in enumerate(y):
            top_3_correct += 1 if (target in top3_preds[idx]) else 0

        total += len(x)

        hidden = model.detach_hidden(hidden)

        # Advance to the next batch
        sample, file_changed = test_dataset_batcher.get_batch()

    print("Next-Token Test Set | Accuracy {:.2f} % | Top 3 Accuracy {:.2f} % | Time taken {:.2f} seconds".
          format(correct / total * 100, top_3_correct / total * 100, time.time() - start))


def next_line_prediction_test(model, word_to_idx, device, args):
    model.eval()

    # Get the data
    test_dataset = NextLineCodeDataset(args.test_files, args.data_root, args.seq_length, word_to_idx)
    test_dataset_batcher = NextLineCodeDatasetBatcher(test_dataset, args.batch_size)

    start = time.time()

    correct = 0
    total = 0
    sample, file_changed = test_dataset_batcher.get_batch()
    while sample is not None:

        for idx, current_input in enumerate(sample[0]):

            previous_tokens = current_input.to(device).unsqueeze(0)
            y = sample[1][idx].to(device)
            # Remove padding
            y = y[y != PAD_IDX]

            if file_changed:
                # Feeding one word at a time, so hidden size should be 1.
                hidden = model.init_hidden(1)

            # Feed the input (previous) lines one word at the time
            predictions, hidden = model(previous_tokens, hidden)

            final_output = []
            for _ in range(len(y)):
                predicted_word = torch.argmax(torch.softmax(predictions, dim=1), dim=1).detach()
                final_output.append(predicted_word)
                predictions, hidden = model(predicted_word.unsqueeze(0), hidden)

            final_output = torch.cat(final_output).to(device)
            total += 1
            correct += 1 if torch.equal(final_output, y) else 0

            hidden = model.detach_hidden(hidden)

        # Advance to the next batch
        sample, file_changed = test_dataset_batcher.get_batch()

    print("Next-Line Test Set | Accuracy {:.2f} % | Time taken {:.2f} seconds"
          .format(correct / total * 100, time.time() - start))

