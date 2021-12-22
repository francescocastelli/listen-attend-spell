import torch
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence
from tokenizer import pad_value

def collate_fn_pad(batch):
    melspecs = [d['melspec'] for d in batch]
    y = [d['token_seq'] for d in batch]

    # get sequence lengths
    lengths = torch.tensor([m.shape[0] for m in melspecs])
    # pad the melspecs 
    pad_melspecs = pad_sequence(melspecs, batch_first=True)
    # pad the sequences of tokens
    pad_y = pad_sequence(y, batch_first=True, padding_value=pad_value)

    b = {}
    # insert the stacked melspec in the batch with their original lengths
    b['melspec'], b['lengths'] = pad_melspecs, lengths
    b['token_seq'] = pad_y

    return b #, lengths, mask
