import time

import torch
import torch.nn as nn
import torch.optim as optim

from utils.data_utils import PAD_IDX
from datasets.full_line_dataset import NextLineCodeDataset, NextLineCodeDatasetBatcher


def train(model, word_to_idx, device, args):
    # Get training and validation data
    train_dataset = NextLineCodeDataset(args.train_files, args.data_root, args.seq_length, word_to_idx)
    val_dataset = NextLineCodeDataset(args.val_files, args.data_root, args.seq_length, word_to_idx)
    train_dataset_batcher = NextLineCodeDatasetBatcher(train_dataset, args.batch_size)

    # Describe the model, create the optimizer and criterion to use
    print("The model {}, has {} trainable parameters.".format(model.save_name, model.summary()))
    encoder_optimiser = optim.Adam(model.encoder.parameters(), lr=args.lr)
    decoder_optimiser = optim.Adam(model.decoder.parameters(), lr=args.lr)
    criterion = nn.NLLLoss(ignore_index=PAD_IDX).to(device)

    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()

        epoch_loss = 0

        epoch_start = time.time()
        # Get current training batch
        sample, file_changed = train_dataset_batcher.get_batch()
        while sample is not None:

            encoder_optimiser.zero_grad()
            decoder_optimiser.zero_grad()

            if file_changed:
                encoder_hidden = model.encoder.init_hidden(args.batch_size)

            if len(sample[0]) != args.batch_size:
                encoder_hidden = encoder_hidden[:, :len(sample[0]), :]

            loss = 0

            # Convert inputs to tensors
            inputs = [torch.tensor(a, device=device) for a in sample[0]]
            targets = [torch.tensor(a, device=device) for a in sample[1]]

            # Pad tensors to seq_length
            targets = nn.utils.rnn.pad_sequence(targets, True, PAD_IDX)

            encoder_hidden, _, decoder_logits = model(inputs, targets, encoder_hidden)
            loss += criterion(decoder_logits.transpose(1, 2), targets)

            # Track the running epoch loss
            epoch_loss += loss.item() / len(sample[0])

            # Backprop the loss and update params, use gradient clipping if specified
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            encoder_optimiser.step()
            decoder_optimiser.step()
            encoder_hidden = encoder_hidden.detach()

            # Get the next batch
            sample, file_changed = train_dataset_batcher.get_batch()

        print("Epoch {} | Loss {:.10f} | Time taken {:.2f} seconds"
              .format(epoch, epoch_loss, time.time() - epoch_start))

        # Validate if we need to
        if epoch % args.val_epochs == 0:
            validate(model, val_dataset, criterion, device, args)

        # Reset the batcher
        train_dataset_batcher.reset_batcher()

    print("Done Training, total time taken: ", time.time() - train_start)
    return model


def validate(model, val_dataset, criterion, device, args):
    val_dataset_batcher = NextLineCodeDatasetBatcher(val_dataset, args.batch_size)
    model.eval()

    validation_loss = 0
    total = 0
    correct = 0

    start_time = time.time()

    # Get current batch
    sample, file_changed = val_dataset_batcher.get_batch()
    while sample is not None:

        if file_changed:
            encoder_hidden = model.encoder.init_hidden(len(sample[0]))

        if len(sample[0]) != args.batch_size:
            encoder_hidden = encoder_hidden[:, :len(sample[0]), :]

        loss = 0

        # Convert inputs to tensors
        inputs = [torch.tensor(a, device=device) for a in sample[0]]
        targets = [torch.tensor(a, device=device) for a in sample[1]]

        # Pad tensors to seq_length
        inputs = nn.utils.rnn.pad_sequence(inputs, True, PAD_IDX)

        encoder_hidden, decoder_outs, decoder_logits = model(inputs, targets, encoder_hidden)

        # Track loss and accuracy
        targets = nn.utils.rnn.pad_sequence(targets, True, PAD_IDX)
        loss += criterion(decoder_logits[:, :targets.size(1),:].transpose(1, 2), targets)
        total += len(targets)
        for idx in range(len(targets)):
            correct += 1 if torch.equal(decoder_outs[idx], targets[idx]) else 0

        encoder_hidden = encoder_hidden.detach()

        validation_loss += loss.item() / len(sample[0])

        # Advance to the next batch
        sample, file_changed = val_dataset_batcher.get_batch()

    print("Validation epoch | Loss {:.10} | Accuracy {:.2f}% | Time taken {:.2f} seconds"
          .format(validation_loss, (correct / total * 100), time.time() - start_time))
