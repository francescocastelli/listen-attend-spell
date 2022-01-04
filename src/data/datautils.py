import torch
import numpy as np
import random
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence
from utils.tokenizer import pad_value, eos_value

# define the collate_fn_pad function
def collate_fn_pad(batch):
    melspecs = [d['melspec'] for d in batch]
    y_in = [d['y_in'] for d in batch]
    y_out = [d['y_out'] for d in batch]

    # get sequence lengths
    lengths = torch.tensor([m.shape[0] for m in melspecs])
    # pad the melspecs 
    pad_melspecs = pad_sequence(melspecs, batch_first=True)
    # pad the sequences of tokens
    pad_y_in = pad_sequence(y_in, batch_first=True, padding_value=eos_value)
    pad_y_out = pad_sequence(y_out, batch_first=True, padding_value=pad_value)

    b = {}
    # insert the stacked melspec in the batch with their original lengths
    b['melspec'], b['lengths'] = pad_melspecs, lengths
    b['y_in'], b['y_out'] = pad_y_in, pad_y_out

    return b #, lengths, mask

# sampler
class SamplerBlockShuffleByLen(torch.utils.data.Sampler):
    def __init__(self, seq_len, batch_size):

        if batch_size == 1:
            raise ValueError(f"Block shuffle sampler requires bs > 1")

        # hyper-parameter
        self.block_size = batch_size * 4
        # num blocks
        self.num_blocks = len(seq_len) // self.block_size 
        # idx sorted based on sequence length
        self.indices = np.argsort(seq_len)

    def __iter__(self):
        tmp_idx = torch.tensor(self.indices)

        # shuffle within each block
        blocks = torch.tensor_split(tmp_idx, self.num_blocks) 
        blocks = [b.view(-1)[torch.randperm(b.numel())].view(b.size()) for b in blocks]

        # shuffle blocks
        random.shuffle(blocks)

        # recreate blocks
        blocks = torch.cat(blocks, dim=0)
        return iter(blocks.tolist())

    def __len__(self):
        return len(self.indices)
