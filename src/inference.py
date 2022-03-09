#!/usr/bin/env python

import os
import torch
import numpy as np
from data.dataset import create_dataset
from frontends.melspectrogram import MelSpectrogram
import utils.parameters as param
from utils.tokenizer import Tokenizer
from model.las import LAS

def main():
    model_path = "/nas/home/fcastelli/asr/src/model/results/las/12-01-22_10:47:20/checkpoints/las_ckpt_epoch_150.pt"
    valid_txt = os.path.join(param.text_path, "train_100.txt")

    tokenizer = Tokenizer()
    model = LAS('las', input_size=64, hidden_size=512, encoder_layers=3,
                decoder_layers=1, embedding_dim=512, 
                tokenizer=tokenizer, inference=True, vocabulary_size=tokenizer.vocabulary_len)

    model_dict = torch.load(model_path, map_location='cuda:0')
    model.load_state_dict(model_dict['model_state_dict'])
    model.eval()

    # frontend
    win_len_s=0.025
    hop_len_s=0.010
    sr=16000
    f_min=20
    f_max=7500
    pad=0
    n_mels=64
    power=2

    win_len=int(round(sr*win_len_s))
    hop_len=int(round(sr*hop_len_s))
    n_fft = 2 ** int(np.ceil(np.log(win_len) / np.log(2.0)))
    frontend = MelSpectrogram(sr=sr, norm=None, n_fft=n_fft, 
                              win_len=win_len, hop_len=hop_len, 
                              f_min=f_min, f_max=f_max, pad=pad, 
                              n_mels=n_mels, window_fn=torch.hann_window, 
                              power=power, normalized=False)


    valid_dataset, _ = create_dataset(valid_txt, param.dataset_dir, tokenizer, frontend)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1)
    valid_it = iter(valid_loader)

    sample = next(valid_it)
    melspec, y_in, y_out = sample['melspec'], sample['y_in'], sample['y_out']
    seq_len = torch.tensor(melspec.shape[1])
    seq_len = torch.unsqueeze(seq_len, dim=0)

    pred_seq = model.inference_step(melspec, seq_len, beam_width=10)

    _, true_seq = tokenizer.decode_tokens(y_out.tolist()[0])
    print(f'ground truth: {true_seq}')

    for i, s in enumerate(pred_seq): 
        print(f'predicted {i}: {s}')

if __name__ == '__main__':
    main()
