import os
import time
import torch
import torch.nn as nn
from code_dataset import CodeDataset, CodeDatasetBatcher


def next_token_prediction_test(model, word_to_idx, device, args):
    # Load the model and set it to eval mode.
    model.load_state_dict(torch.load(os.path.join(args.model_path, model.save_name)))
    model.eval()

    # Get the data
    test_dataset = CodeDataset(args.test_files, args.data_root, args.seq_length, word_to_idx)
    test_dataset_batcher = CodeDatasetBatcher(test_dataset, args.batch_size)

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
