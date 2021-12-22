import torch
import numpy as np
import os
import parameters as param
from dataset import create_dataset
from frontends.melspectrogram import MelSpectrogram
from tokenizer import Tokenizer
from datautils import collate_fn_pad
from model import LAS 
from torchtrainer.trainer import Trainer 
from torchtrainer.dataloader import TrainerLoader
from argparser import parse_args

def main(args):
    train_txt = os.path.join(param.text_path, "train.txt")
    valid_txt = os.path.join(param.text_path, "dev.txt")

    # frontend
    win_len=int(round(args.sr*args.win_len_s))
    hop_len=int(round(args.sr*args.hop_len_s))
    n_fft = 2 ** int(np.ceil(np.log(win_len) / np.log(2.0)))
    frontend = MelSpectrogram(sr=args.sr, norm=None, n_fft=n_fft, 
                             win_len=win_len, hop_len=hop_len, 
                             f_min=args.f_min, f_max=args.f_max, pad=args.pad, 
                             n_mels=args.n_mels, window_fn=torch.hann_window, 
                             power=args.power, normalized=False)

    # tokenizer
    tokenizer = Tokenizer()

    # dataset
    train_dataset = create_dataset(train_txt, param.dataset_dir, tokenizer, frontend)
    valid_dataset = create_dataset(valid_txt, param.dataset_dir, tokenizer, frontend)

    # model
    model = LAS(args.name, 64, 512, 
                512, tokenizer.vocabulary_len, args)

    # dataloader
    dataloader = TrainerLoader(batch_size=args.bs, collate_fn=collate_fn_pad,
                              num_workers=args.w, shuffle=True)

    trainer = Trainer(model=model, train_dataset=train_dataset, 
                     valid_dataset=valid_dataset, 
                     loader=dataloader, epoch_num=args.epochs, summary_args=vars(args), 
                     distributed=args.distributed, device=args.d,
                     print_stats=args.console_logs, tb_logs=args.logs, 
                     tb_embeddings=args.embeddings, save_path=param.log_dir, 
                     seed=args.seed) 

    if args.multitrain:
        trainer.multi_train(param.train_config_path)
    else: 
        trainer.train()

if __name__ == '__main__':
    args = parse_args() 
    main(args)
