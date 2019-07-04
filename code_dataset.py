import os
import tokenize
import itertools
import numpy as np
import data_utils as du
from torch.utils.data import Dataset


class CodeDataset(Dataset):

    def __init__(self, txt_file, root_folder, sequence_length, vocabulary):
        """
        :param txt_file: File with the relative paths to the files in this dataset.
        :param root_folder: Folder containing the .py files.
        :param sequence_length: Max length for the sequences of tokens.
        :param vocabulary: word to int dictionary mapping.
        """
        self.root_dir = root_folder
        self.seq_length = sequence_length
        self.vocabulary = vocabulary
        self.code_files = []

        with open(txt_file, 'r') as data_db:
            self.code_files = [x for x in data_db.read().splitlines()]

    def __len__(self):
        # Need to return the length of the dataset somehow.
        # The code below might not be the best option.
        return len(self.code_files)

    def __getitem__(self, idx):
        filename = self.code_files[idx]
        print("Loading File:", filename)
        sample_tokens = self.__obtain_tokens(filename)
        sample_tokens = self.__vecotrize_and_pad(sample_tokens)

        prepared_outputs = np.zeros_like(sample_tokens)
        prepared_outputs[:-1] = sample_tokens[1:]
        prepared_outputs[-1] = sample_tokens[0]

        return sample_tokens, prepared_outputs

    def __obtain_tokens(self, filename):
        sample = []
        with open(os.path.join(self.root_dir, filename), 'r') as current_file:
            tokens = tokenize.generate_tokens(current_file.readline)

            # Dont process comments, newlines, block comments or empty tokens
            processed_tokens = [du.preprocess(t_type, t_val) for t_type, t_val, _, _, _ in tokens
                                if t_type != tokenize.COMMENT and
                                not t_val.startswith("'''") and
                                not t_val.startswith('"""') and
                                (t_type == tokenize.DEDENT or t_val != "")]
            if processed_tokens:
                sample.append(processed_tokens)
        return sample

    def __vecotrize_and_pad(self, token_list):
        result = []
        token_list = np.array(list(itertools.chain(*token_list)))
        newlines = np.where(token_list == '\n')[0] if token_list.size > 0 else []

        start_idx = 0
        for newline_idx in newlines:
            if start_idx != newline_idx:
                current_line = [self.vocabulary.get(word, du.OOV_IDX) for word in token_list[start_idx:newline_idx][:self.seq_length]]
                current_line = np.pad(current_line, (0, self.seq_length - len(current_line)), mode='constant',
                                      constant_values=du.PAD_IDX)

                result.append(current_line)
                start_idx = newline_idx + 1

        return np.array(result)


class CodeDatasetBatcher:

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.current_file = 0
        self.current_position = 0

    def reset_batcher(self):
        self.current_file = 0
        self.current_position = 0

    def get_batch(self):
        if self.current_file >= len(self.dataset):
            return None
        try:
            inputs, outputs = self.dataset[self.current_file]
        except IndexError:
            # Handle empty files as well by advancing in the file list
            self.current_file += 1
            self.current_position = 0
            return self.get_batch()

        if len(inputs) >= self.batch_size and self.current_position + self.batch_size <= len(inputs):
            result = inputs[self.current_position:self.current_position + self.batch_size],\
                  outputs[self.current_position:self.current_position + self.batch_size]
            self.current_position += self.batch_size

        else:
            result = inputs[self.current_position:], outputs[self.current_position:]
            self.current_file += 1
            self.current_position = 0

        return result
