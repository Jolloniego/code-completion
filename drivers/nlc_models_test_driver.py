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
        if file_changed:
            encoder_hidden = model.encoder.init_hidden()

        for idx, encoder_input in enumerate(sample[0]):
            encoder_input = torch.tensor(encoder_input, device=device)
            target_tensor = torch.tensor(sample[1][idx], device=device).unsqueeze(0)

            _, encoder_hidden, decoder_outs = model(encoder_input, target_tensor, encoder_hidden)

            # Track accuracy
            total += 1
            correct += 1 if torch.equal(target_tensor, decoder_outs) else 0

            encoder_hidden = encoder_hidden.detach()

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

        if file_changed:
            encoder_hidden = model.encoder.init_hidden()

        for idx, encoder_input in enumerate(sample[0]):
            encoder_input = torch.tensor(encoder_input, device=device)
            target_tensor = torch.tensor(sample[1][idx], device=device)

            _, encoder_hidden, decoder_outs = model(encoder_input, target_tensor, encoder_hidden)

            # Track accuracy
            total += 1
            correct += 1 if torch.equal(target_tensor, decoder_outs) else 0

            encoder_hidden = encoder_hidden.detach()
        # Advance to the next batch
        sample, file_changed = test_dataset_batcher.get_batch()

    print("Next-Line Test Set | Accuracy {:.2f} % | Time taken {:.2f} seconds"
          .format(correct / total * 100, time.time() - start))
