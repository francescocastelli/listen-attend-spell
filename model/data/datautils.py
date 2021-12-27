import torch
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence
from tokenizer import pad_value

def collate_fn_pad(batch):
    melspecs = [d['melspec'] for d in batch]
    y_in = [d['y_in'] for d in batch]
    y_out = [d['y_out'] for d in batch]

    # get sequence lengths
    lengths = torch.tensor([m.shape[0] for m in melspecs])
    # pad the melspecs 
    pad_melspecs = pad_sequence(melspecs, batch_first=True)
    # pad the sequences of tokens
    pad_y_in = pad_sequence(y_in, batch_first=True, padding_value=pad_value)
    pad_y_out = pad_sequence(y_out, batch_first=True, padding_value=pad_value)

    b = {}
    # insert the stacked melspec in the batch with their original lengths
    b['melspec'], b['lengths'] = pad_melspecs, lengths
    b['y_in'], b['y_out'] = pad_y_in, pad_y_out

    return b #, lengths, mask
