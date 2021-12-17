#!/usr/bin/env python3

import os
import glob
import re
import pandas as pd
import argparse

def get_target_info(target_name):
    r = re.split('\-|\.', target_name)
    return r[0], r[1]

def get_target_seq_info(target_seq, dataset):
    r = target_seq.split(' ', 1)
    data = re.split('\-', r[0])
    return [data[0], data[1], data[2], r[1], dataset]

def compute_df(target_dir: list, dataset: list):
    assert len(target_dir) == len(dataset), "target_dir and dataset must have the same len"

    target_seq = []
    for t_dir, d in zip(target_dir, dataset):
        t_list = [os.path.basename(path) for path in glob.glob(os.path.join(t_dir, "*", "*", "*.txt"))]

        for t in t_list:
            speaker, chap = get_target_info(t)

            with open(os.path.join(t_dir, speaker, chap, t)) as f:
                lines = f.readlines()
                lines = [get_target_seq_info(l, d) for l in lines]

            target_seq.extend(lines)

    df = pd.DataFrame(target_seq, columns=['speaker_id', 'chapter_id', 'audio_id', 'seq', 'dataset_id'])
    return df 


def main(args):
    dataset_dir = "/nas/public/dataset/LibriSpeech"
    text_path = "/nas/home/fcastelli/asr/dataset/librispeech/texts"

    out_path = os.path.join(text_path, args.text_name)
    dataset_names = [name for name in args.dataset_names]
    train_dir = [os.path.join(dataset_dir, t_dir) for t_dir in dataset_names]

    final_df = compute_df(train_dir, dataset_names)
    final_df.to_csv(out_path, sep=' ', header=False, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_name", type=str, help="name of the ouput file")
    parser.add_argument("--dataset_names", type=str, nargs="*", help="list of string, where each one is a dataset dir name (eg. train-clean-100)")
    args = parser.parse_args()
    main(args)
