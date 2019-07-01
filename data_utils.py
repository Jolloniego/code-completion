import os
import tokenize

STR_TOKEN = '<STR>'
NUM_TOKEN = '<NUM>'
INDENT_TOKEN = '<IND>'
DEDENT_TOKEN = '<DED>'
OOV_TOKEN, OOV_IDX = '<OOV>', 0
PAD_TOKEN, PAD_IDX = '<PAD>', 1


def read_data(data_file_path, data_root):
    data = []
    with open(data_file_path, 'r') as train_db:
        python_files = [x for x in train_db.read().splitlines()]

    # Read each code file to be used from the specified path
    for filename in python_files:
        try:
            with open(os.path.join(data_root, filename), 'r') as current_file:
                tokens = tokenize.generate_tokens(current_file.readline)

                # Dont process comments, newlines, block comments or empty tokens
                data.append([preprocess(t_type, t_val) for t_type, t_val, _, _, _ in tokens
                             if t_type != tokenize.COMMENT and
                             not t_val.startswith("'''") and
                             not t_val.startswith('"""') and
                             # not t_val == '\n' and
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
