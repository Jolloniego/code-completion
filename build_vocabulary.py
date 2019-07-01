import pickle
import argparse
import itertools
import collections
import data_utils as du

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--vocab_data', type=str, help='Path to file containing paths to the data to use.')
parser.add_argument('--data_root', default='data', type=str, help='Path root folder containing the cloned repositories.')
parser.add_argument('--out_path', default='data', type=str, help='Path to save vocabulary object.')
args = parser.parse_args()


def build_vocab(dataset):
    oov_threshold = 0
    counter = collections.Counter(itertools.chain(itertools.chain(*dataset)))
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])

    count_pairs = (p for p in count_pairs if p[1] > oov_threshold)

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(2, len(words) + 2)))
    word_to_id[du.OOV_TOKEN] = du.OOV_IDX
    word_to_id[du.PAD_TOKEN] = du.PAD_IDX
    return word_to_id


if __name__ == "__main__":
    data_for_vocab = args.vocab_data
    data_root = args.data_root
    data = du.read_data(data_for_vocab, data_root)
    print("Loaded {} files".format(len(data)))
    vocab = build_vocab(data)
    print("Vocabulary size is:", len(vocab))
    out_file = args.out_path + "/vocab.p"
    pickle.dump(vocab, open(out_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
