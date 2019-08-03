import time

import torch
import torch.nn as nn

from utils.data_utils import PAD_IDX
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
            encoder_hidden = model.encoder.init_hidden(len(sample[0]))

        encoder_input = torch.tensor(sample[0], device=device)
        target_tensor = torch.tensor(sample[1], device=device).unsqueeze(1)

        encoder_hidden, decoder_logits = model(encoder_input, target_tensor, encoder_hidden)

        predictions = decoder_logits[:, :1, :].topk(1)[1].squeeze()
        del decoder_logits

        # Track accuracy
        total += len(sample[0])
        correct += torch.sum(torch.eq(predictions, target_tensor.squeeze())).item()

        encoder_hidden = encoder_hidden.detach()

        # Advance to the next batch
        sample, file_changed = test_dataset_batcher.get_batch()

    print("Next-Token Test Set | Accuracy {:.2f} % | Time taken {:.2f} seconds".format(correct / total * 100, time.time() - start))


def next_line_prediction_test(model, word_to_idx, device, model_path, args):
    # Load the model and set it to eval mode.
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Get the data
    test_dataset = NextLineCodeDataset(args.test_files, args.data_root, args.seq_length, word_to_idx)
    test_dataset_batcher = NextLineCodeDatasetBatcher(test_dataset, args.batch_size)

    start = time.time()

    correct = 0
    total = 0
    sample, file_changed = test_dataset_batcher.get_batch()
    while sample is not None:

        if file_changed:
            encoder_hidden = model.encoder.init_hidden(len(sample[0]))

        if len(sample[0]) != args.batch_size:
            encoder_hidden = encoder_hidden[:, :len(sample[0]), :]

        targets = sample[1]

        # Pad into single tensor
        inputs = nn.utils.rnn.pad_sequence(sample[0], True, PAD_IDX).to(device)

        encoder_hidden, decoder_logits = model(inputs, targets, encoder_hidden)

        # Free some memory
        del inputs

        # Convert logits into token predictions and free memory
        token_predictions = decoder_logits.topk(1)[1].squeeze().cpu()
        del decoder_logits

        total += len(targets)
        for idx in range(len(targets)):
            correct += 1 if torch.equal(token_predictions[idx][:len(targets[idx])], targets[idx]) else 0

        encoder_hidden = encoder_hidden.detach()

        # Advance to the next batch
        sample, file_changed = test_dataset_batcher.get_batch()

    print("Next-Line Test Set | Accuracy {:.2f} % | Time taken {:.2f} seconds"
          .format(correct / total * 100, time.time() - start))
