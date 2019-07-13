import time
import torch
import torch.nn as nn
import torch.optim as optim
from code_dataset import CodeDataset, CodeDatasetBatcher


def train(model, word_to_idx, device, args):
    # Get training and validation data
    train_dataset = CodeDataset(args.train_files, args.data_root, args.seq_length, word_to_idx)
    val_dataset = CodeDataset(args.val_files, args.data_root, args.seq_length, word_to_idx)
    train_dataset_batcher = CodeDatasetBatcher(train_dataset, args.batch_size)

    # Create the model, optimizer and criterion to use
    print("The model has {} trainable parameters.".format(model.summary()))
    optimiser = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().to(device)

    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()

        epoch_loss = 0
        correct = 0
        total = 0

        epoch_start = time.time()
        # Get current training batch
        sample, file_changed = train_dataset_batcher.get_batch()
        while sample is not None:
            optimiser.zero_grad()

            x = torch.tensor(sample[0], device=device)
            y = torch.tensor(sample[1], device=device)

            if file_changed:
                hidden = model.init_hidden(len(x))

            # Get the predictions and compute the loss
            preds, hidden = model(x, hidden)
            loss = criterion(preds, y)

            # Track accuracy as well
            total += len(x)
            preds = torch.argmax(nn.functional.softmax(preds, dim=1), dim=1).detach()
            correct += (preds == y).sum().item()

            # Backprop the loss and update params, use gradient clipping if specified
            loss.backward(retain_graph=True)
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimiser.step()

            epoch_loss += loss.item() / len(x)

            # Get the next batch
            sample, file_changed = train_dataset_batcher.get_batch()

        print("Epoch {} | Loss {:.10} | Accuracy {:.2f}% | Time taken {:.2f} seconds"
              .format(epoch, epoch_loss, (correct / total * 100), time.time() - epoch_start))

        # Validate if we need to
        if epoch % args.val_epochs == 0:
            model.eval()
            validate(model, val_dataset, time.time(), device, args)

        # Reset the batcher
        train_dataset_batcher.reset_batcher()

    print("Done Training, total time taken: ", time.time() - train_start)
    return model


def validate(model, val_dataset, start_time, device, args):
    val_dataset_batcher = CodeDatasetBatcher(val_dataset, args.batch_size)
    criterion = nn.CrossEntropyLoss().to(device)

    validation_loss = 0
    total = 0
    correct = 0

    sample, file_changed = val_dataset_batcher.get_batch()
    while sample is not None:
        x = torch.tensor(sample[0], device=device)
        y = torch.tensor(sample[1], device=device)

        if file_changed:
            hidden = model.init_hidden(len(x))

        preds, hidden = model(x, hidden)
        loss = criterion(preds, y).item()
        validation_loss += loss / len(x)

        # Track accuracy
        total += len(x)
        preds = torch.argmax(nn.functional.softmax(preds, dim=1), dim=1).detach()
        correct += (preds == y).sum().item()

        # Advance to the next batch
        sample, file_changed = val_dataset_batcher.get_batch()

    print("Validation epoch | Loss {:.10} | Accuracy {:.2f}% | Time taken {:.2f} seconds"
          .format(validation_loss, (correct / total * 100), time.time() - start_time))
