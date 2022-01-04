import string
import torch
from collections import defaultdict

pad_value = 0
eos_value = 1
sos_value = 2

class Tokenizer():
    r"""
    Tokenizer contains a dictionary of tokens, where each token is a character. In particular we consider:
        * digits
        * all letters of the english alphabet 
        * punction: dot, comma, colon, space
        * sos: start of the sentence
        * eos: end of the sentence
        * pad: special token used to pad sequences (set to 0)
        * unk: all the other characters
    """
    def __init__(self):
        # tokens = lower-case lettes, numbers and some punctuation
        tokens = [*string.ascii_lowercase, *range(0, 10), ' ', ',', '.', '\'', ':']

        self.token_dict = defaultdict(lambda: len(tokens), {c: k for k, c in enumerate(tokens, 3)})
        self.char_dict = {v: k for k, v in self.token_dict.items()}

        # char not in dict gets len(tokens) -> unk
        self.char_dict[len(tokens)] = 'unk'

        # set constant special tokens
        self.char_dict[pad_value] = 'pad'
        self.char_dict[sos_value] = 'sos'
        self.char_dict[eos_value] = 'eos'

        self.vocabulary_len = len(self.char_dict)


    def tokenize(self, seq: list):
        r"""
           Used to tokenized the input sequence.

           Args:
                seq: list of char that compose the sequence

           Returns: 
                list of tokens corresponding to the input seq,
                with sos and eos at the beginning and end
        """
        token_seq = [self.token_dict[c.lower()] for c in seq]
        token_seq.insert(0, sos_value)
        token_seq.append(eos_value)

        return token_seq

    def decode_tokens(self, token_seq: list):
        r"""
           Reconstruct the seq starting from a list of tokens

           Args:
                token_seq: list of int that compose the token sequence

           Returns: 
                list of chars corresponding to the token seq
        """
        char_list = [self.char_dict[t] for t in token_seq if t != sos_value and t != eos_value]
        string = "".join(char_list)

        return char_list, string
