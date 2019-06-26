import pickle
import tokenize
import itertools
import collections

TRAIN_DATA = 'data/train.txt'

STR_TOKEN = '<STR>'
NUM_TOKEN = '<NUM>'
INDENT_TOKEN = '<IND>'
DEDENT_TOKEN = '<DED>'
OOV_TOKEN, OOV_IDX = '<OOV>', 0
PAD_TOKEN, PAD_IDX = '<PAD>', 1


def read_data(data_path):
    data = []
    with open(data_path, 'r') as train_db:
        python_files = [x for x in train_db.read().splitlines()]

    # Read each code file to be used from the specified path
    for filename in python_files:
        try:
            with open(filename, 'r') as current_file:
                tokens = tokenize.generate_tokens(current_file.readline)

                # Dont process comments, block comments or empty tokens
                data.append([preprocess(t_type, t_val) for t_type, t_val, _, _, _ in tokens
                             if t_type != tokenize.COMMENT and
                             not t_val.startswith("'''") and
                             not t_val.startswith('"""') and
                            (t_type == tokenize.DEDENT or t_val != "")])

        except:
            print("Error with file:", filename)

    return data


def preprocess(token_type, token_val):
    if token_type == tokenize.NUMBER:
        return NUM_TOKEN
    if token_type == tokenize.INDENT:
        return INDENT_TOKEN
    if token_type == tokenize.DEDENT:
        return DEDENT_TOKEN
    if token_type == tokenize.STRING:
        return STR_TOKEN
    return token_val


def build_vocab(data):
    oov_threshold = 0
    counter = collections.Counter(itertools.chain(itertools.chain(*data)))
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])

    count_pairs = (p for p in count_pairs if p[1] > oov_threshold)

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(2, len(words) + 2)))
    word_to_id[OOV_TOKEN] = OOV_IDX
    word_to_id[PAD_TOKEN] = PAD_IDX
    return word_to_id


if __name__ == "__main__":
    data = read_data('data/train.txt')
    print(len(data))
    vocab = build_vocab(data)
    print(len(vocab))
    pickle.dump(vocab, open("vocab.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
