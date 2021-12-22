import string
import torch
from collections import defaultdict

pad_value = 0

class Tokenizer():
    def __init__(self):
        # tokens = lower-case lettes, numbers and some punctuation
        tokens = [*string.ascii_lowercase, *range(0, 10), ' ', ',', '.', '\'', ':', 'sos', 'eos']

        self.token_dict = defaultdict(lambda: tokens_num, {c: k for k, c in enumerate(tokens, 1)})
        self.char_dict = {v: k for k, v in self.token_dict.items()}
        
        # char not in dict gets len(tokens) -> unk
        self.char_dict[len(tokens)] = 'unk'
        # pad token -> len(char_dict) 
        self.char_dict[pad_value] = 'pad'

        self.vocabulary_len = len(self.char_dict)

    def tokenize(self, target_seq: list):
        token_seq = [self.token_dict[c.lower()] for c in target_seq]
        token_seq.insert(0, self.token_dict['sos'])
        token_seq.append(self.token_dict['eos'])

        return token_seq

    def decode_tokens(self, token_seq: list):
        char_list = [self.char_dict[t.item()] for t in token_seq]
        string = "".join(char_list[1:-1])

        return char_list, string
    
