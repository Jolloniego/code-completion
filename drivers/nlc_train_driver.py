import time

import torch
import torch.nn as nn
import torch.optim as optim

from datasets.full_line_dataset import NextLineCodeDataset, NextLineCodeDatasetBatcher

from models.baseline_seq2seq import BaselineEncoderDecoderModel


def train(model, word_to_idx, device, args):
    # Get training and validation data
    train_dataset = NextLineCodeDataset(args.train_files, args.data_root, args.seq_length, args.prev_lines, word_to_idx)
    val_dataset = NextLineCodeDataset(args.val_files, args.data_root, args.seq_length, args.prev_lines, word_to_idx)
    train_dataset_batcher = NextLineCodeDatasetBatcher(train_dataset, args.batch_size)

    # Create the model, optimizer and criterion to use
    model = BaselineEncoderDecoderModel(len(word_to_idx), 300, args.dropout, device)

    print("The model {}, has {} trainable parameters.".format(model.save_name, model.summary()))
    encoder_optimiser = optim.Adam(model.encoder.parameters(), lr=args.lr)
    decoder_optimiser = optim.Adam(model.decoder.parameters(), lr=args.lr)
    criterion = nn.NLLLoss().to(device)

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
                encoder_hidden = model.encoder.init_hidden()

            loss = 0

            for idx, encoder_input in enumerate(sample[0]):
                #encoder_hidden = model.encoder.init_hidden()

                encoder_input = torch.tensor(encoder_input, device=device)
                target_tensor = torch.tensor(sample[1][idx], device=device)

                current_loss, encoder_hidden = model(encoder_input, target_tensor, encoder_hidden, criterion)
                loss += current_loss

            # Track the running epoch loss
            epoch_loss += loss.item() / len(sample[0])

            # Backprop the loss and update params, use gradient clipping if specified
            loss.backward()#retain_graph=True)
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            encoder_optimiser.step()
            decoder_optimiser.step()

            # Get the next batch
            sample, file_changed = train_dataset_batcher.get_batch()

        print("Epoch {} | Loss {:.10f} | Time taken {:.2f} seconds"
              .format(epoch, epoch_loss, time.time() - epoch_start))

        # Validate if we need to
        if epoch % args.val_epochs == 0:
            model.eval()
            validate(model, val_dataset, time.time(), device, args)

        # Reset the batcher
        train_dataset_batcher.reset_batcher()

    print("Done Training, total time taken: ", time.time() - train_start)
    return model


def validate(model, val_dataset, start_time, device, args):
    val_dataset_batcher = NextLineCodeDatasetBatcher(val_dataset, args.batch_size)
    criterion = nn.CrossEntropyLoss().to(device)

    validation_loss = 0
    total = 0
    correct = 0

    sample, file_changed = val_dataset_batcher.get_batch()
    while sample is not None:

        if file_changed:
            hidden = model.init_hidden(args.seq_length)

        for idx, current_input in enumerate(sample[0]):

            x = torch.tensor(current_input, device=device)
            y = torch.tensor(sample[1][idx], device=device)

            preds, hidden = model(x, hidden)
            loss = criterion(preds, y).item()
            validation_loss += loss / len(x)

            # Track accuracy
            total += 1
            preds = torch.argmax(nn.functional.softmax(preds, dim=1), dim=1).detach()
            correct += 1 if torch.equal(preds, y) else 0

        # Advance to the next batch
        sample, file_changed = val_dataset_batcher.get_batch()

    print("Validation epoch | Loss {:.10} | Accuracy {:.2f}% | Time taken {:.2f} seconds"
          .format(validation_loss, (correct / total * 100), time.time() - start_time))
