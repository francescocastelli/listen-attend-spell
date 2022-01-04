import os
import torch
import numpy as np
from torchtrainer.trainer import Trainer 
from torchtrainer.dataloader import TrainerLoader
from model.las import LAS 
from frontends.melspectrogram import MelSpectrogram
from data.dataset import create_dataset
from data.datautils import collate_fn_pad, SamplerBlockShuffleByLen
from utils.tokenizer import Tokenizer
from utils.argparser import parse_args
import utils.parameters as param

def main(args):
    train_txt = os.path.join(param.text_path, "train_100.txt")
    valid_txt = os.path.join(param.text_path, "dev.txt")

    ckpt_path = None
    if args.ckpt:
        date = "01-01-22_12:35:11"
        ckpt_path = os.path.join(param.log_dir, args.name, date, args.name+'.pt')

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
    train_dataset, train_len = create_dataset(train_txt, param.dataset_dir, tokenizer, frontend)
    valid_dataset, valid_len = create_dataset(valid_txt, param.dataset_dir, tokenizer, frontend)

    # model
    model = LAS(args.name, input_size=64, hidden_size=512, encoder_layers=3, 
                decoder_layers=1, embedding_dim=512, 
                vocabulary_size=tokenizer.vocabulary_len, 
                tokenizer=tokenizer, args=args)

    # dataloader
    train_sampler = SamplerBlockShuffleByLen(train_len, args.bs)
    valid_sampler = SamplerBlockShuffleByLen(valid_len, args.bs)
    dataloader = TrainerLoader(batch_size=args.bs, collate_fn=collate_fn_pad,
                               train_sampler=train_sampler, valid_sampler=valid_sampler,
                               num_workers=args.w, shuffle=True)

    trainer = Trainer(model=model, train_dataset=train_dataset, 
                     valid_dataset=valid_dataset, 
                     loader=dataloader, epoch_num=args.epochs, summary_args=vars(args), 
                     distributed=args.distributed, device=args.d,
                     verbose=args.console_logs, tb_logs=args.logs, 
                     tb_embeddings_num=args.embeddings_num, results_path=param.log_dir, 
                     seed=args.seed, tb_checkpoint_rate=args.ckpt_rate,
                     checkpoint_path=ckpt_path)

    if args.multitrain:
        trainer.multi_train(param.train_config_path)
    else: 
        trainer.train()

if __name__ == '__main__':
    args = parse_args() 
    main(args)
