import os
import tokenize

import numpy as np
from torch.utils.data import Dataset

from utils import data_utils as du


class NextLineCodeDataset(Dataset):
    def __init__(self, txt_file, root_folder, sequence_length, previous_lines, vocabulary):
        """
        :param txt_file: File with the relative paths to the files in this dataset.
        :param root_folder: Folder containing the .py files.
        :param sequence_length: Max length for the sequences of tokens.
        :param previous_lines: Previous lines to create the input.
        :param vocabulary: word to int dictionary mapping.
        """
        self.root_dir = root_folder
        self.seq_length = sequence_length
        self.previous_lines = previous_lines
        self.vocabulary = vocabulary
        self.code_files = []

        # List of tuples with already loaded inputs and outputs to speed up epochs after the 1st
        self.loaded_files = []

        with open(txt_file, 'r', encoding="utf-8") as data_db:
            self.code_files = [x for x in data_db.read().splitlines()]

    def __len__(self):
        return len(self.code_files)

    def __getitem__(self, idx):
        """
        Loads the selected file and processes it, it then creates and shapes the inputs and output paris.
        Saves the result to speed up future use.
        :param idx: int. Index of file to get.
        :return: Tuple of (inputs, outputs) from the selected file.
        """
        try:
            return self.loaded_files[idx]

        except IndexError:
            # Process each file only once.
            filename = self.code_files[idx]
            sample_tokens = [line for line in self.__obtain_tokens(filename) if line != []]

            prepared_outputs = []
            prepared_inputs = []
            position = 0
            while position < len(sample_tokens):
                current_sample = np.concatenate(sample_tokens[position:position + self.previous_lines + 1])
                if len(current_sample) > self.seq_length:
                    prepared_outputs.append(current_sample[-self.seq_length:])
                    current_sample = current_sample[:-self.seq_length]

                    amount_to_pad = 0 if current_sample.size % self.seq_length == 0 else \
                        abs(current_sample.size - ((current_sample.size // self.seq_length) + 1) * self.seq_length)
                    current_sample = np.pad(current_sample, (amount_to_pad, 0), mode='constant',
                                            constant_values=du.PAD_IDX)

                    prepared_inputs.append(current_sample.reshape(self.seq_length, -1))
                position += self.previous_lines + 1

            self.loaded_files.append((prepared_inputs, prepared_outputs))

            return self.loaded_files[idx]

    def __obtain_tokens(self, filename):
        """
        Opens the selected file and returns a list of tokens for it using the tokenize library.
        :param filename: path to file to be opened.
        :return: List of tokens (or empty list in case of Error)
        """
        sample = []
        try:
            current_file = open(os.path.join(self.root_dir, filename), 'r', encoding='utf-8')
            tokens = tokenize.generate_tokens(current_file.readline)

            # Dont process comments, newlines, block comments or empty tokens
            processed_tokens = [du.preprocess(t_type, t_val) for t_type, t_val, _, _, _ in tokens
                                if t_type != tokenize.COMMENT and
                                not t_val.startswith("'''") and
                                not t_val.startswith('"""') and
                                (t_type == tokenize.DEDENT or t_val != "")]
            if processed_tokens:
                line_by_line = []
                all_tokens = np.array(processed_tokens)
                newlines = np.where(all_tokens == '\n')[0]
                current_pos = 0
                for newline_pos in newlines:
                    line = all_tokens[current_pos:newline_pos + 1]
                    current_pos = newline_pos + 1
                    line_by_line.append(self.__vectorize(line))

                sample.extend(line_by_line)
        except OSError:
            pass
        return sample

    def __vectorize(self, token_list):
        """
        Converts each word to a one-hot vector.
        :param token_list: List of tokens returned from __obtain_tokens.
        :return: ndarray [n_samples, seq_len + 1].
        """
        current_sequence = np.array([self.vocabulary.get(word, du.OOV_IDX) for word in token_list], dtype=np.long)
        return current_sequence


class NextLineCodeDatasetBatcher:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.current_file = 0
        self.changed_file = True
        self.current_position = 0

    def reset_batcher(self):
        """
        Resets the batcher to the starting position.
        """
        self.current_file = 0
        self.current_position = 0
        self.changed_file = True

    def get_batch(self):
        """
        Performs the necessary computations to generate a new batch from the data.
        Returns None if dataset has been explored entirely.
        :return: Current batch to be processed by the models.
        """
        if self.current_file >= len(self.dataset):
            return None, self.changed_file

        inputs, outputs = self.dataset[self.current_file]

        if len(inputs) >= self.batch_size and self.current_position + self.batch_size <= len(inputs):
            result = inputs[self.current_position:self.current_position + self.batch_size], \
                     outputs[self.current_position:self.current_position + self.batch_size]
            self.changed_file = self.current_position == 0
            self.current_position += self.batch_size

        else:
            result = inputs[self.current_position:], outputs[self.current_position:]
            self.current_file += 1
            self.changed_file = True
            self.current_position = 0

        if len(result[0]) == 0 or len(result[0][0]) == 0:
            return self.get_batch()

        return result, self.changed_file