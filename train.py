import model
import torch
import pickle
import numpy as np
import torch.nn as nn
import data_utils as du
import torch.optim as optim

TRAINING_DATA_PATH = 'data/train.txt'
BATCH_SIZE = 32
SEQ_LEN = 10
NUM_EPOCHS = 30

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


word_to_idx = pickle.load(open('vocab.p', 'rb'))
ixd_to_word = {key: word for key, word in enumerate(word_to_idx)}


def prepare_data(data_path, batch_size, seq_len):
    data = du.read_data(data_path)

    text_to_int = np.array([[word_to_idx[word] for word in file] for file in data if file != []]).flatten()

    num_batches = int(len(text_to_int) / (seq_len * batch_size))
    prepared_inputs = text_to_int[:num_batches * batch_size * seq_len]
    prepared_outputs = np.zeros_like(prepared_inputs)
    prepared_outputs[:-1] = prepared_inputs[1:]
    prepared_outputs[-1] = prepared_inputs[0]

    return prepared_inputs.reshape((batch_size, -1)), prepared_outputs.reshape((batch_size, -1))


def generate_batches(inputs, outputs, batch_size, seq_len):
    num_batches = np.prod(inputs.shape) // (seq_len * batch_size)
    for i in range(0, num_batches * seq_len, seq_len):
        yield inputs[:, i:i + seq_len], outputs[:, i:i + seq_len]


ins, outs = prepare_data(TRAINING_DATA_PATH, BATCH_SIZE, SEQ_LEN)

model = model.DummyModel(len(word_to_idx), 300)
criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(NUM_EPOCHS):
    model.train()

    epoch_loss = 0
    for x, y in generate_batches(ins, outs, BATCH_SIZE, SEQ_LEN):

        model.zero_grad()

        x = torch.LongTensor(x)
        y = torch.LongTensor(y)

        # Get the predictions and compute the loss
        preds = model(x)
        loss = criterion(preds.view(-1, 90), y.view(-1))

        # Backprop the loss and update params
        loss.backward()
        optimiser.step()

        epoch_loss += loss.data

    print(epoch_loss)

print("asd")


