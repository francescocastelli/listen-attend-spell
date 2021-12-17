import string
import torch
from collections import defaultdict

class Tokenizer():
    def __init__(self):
        # tokens = lower-case lettes, numbers and some punctuation
        tokens = [*string.ascii_lowercase, *range(0, 10), ' ', ',', '.', '\'', ':', 'sos', 'eos']
        tokens_num = len(tokens)

        # char not in dict gets len(tokens) -> unk
        self.token_dict = defaultdict(lambda: tokens_num, {c: k for k, c in enumerate(tokens)})
        self.char_dict = {v: k for k, v in self.token_dict.items()}
        self.char_dict[tokens_num] = 'unk'

    def tokenize(self, target_seq: list):
        token_seq = [self.token_dict[c.lower()] for c in target_seq]
        token_seq.insert(0, self.token_dict['sos'])
        token_seq.append(self.token_dict['eos'])

        return token_seq

    def decode_tokens(self, token_seq: list):
        char_list = [self.char_dict[t.item()] for t in token_seq]
        string = "".join(char_list[1:-1])

        return char_list, string
