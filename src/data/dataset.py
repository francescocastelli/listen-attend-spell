import torch
import torchaudio
import os
import pandas as pd

class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, df, dataset_dir, tokenizer, frontend):
        self.df = df
        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.frontend = frontend
    
    def __len__(self):
        return len(self.df)
    
    def __get_path(self, item):
        speaker, chapter, dataset, audio = item['speaker_id'], item['chapter_id'], item['dataset'], item['audio_id']
        filename = f"{speaker}-{chapter}-{audio}.flac"
        path = os.path.join(self.dataset_dir, dataset, str(speaker), str(chapter), filename)
        return path
    
    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        
        audio_path = self.__get_path(sample)
        audio, _ = torchaudio.load(audio_path)

        # compute melspec
        melspec = self.frontend(audio[0])
        # shape: (L, n_mels) where L depends on the length in time
        melspec = torch.t(melspec)
        
        # tokenize the target seq
        target = sample['seq']
        token_seq = self.tokenizer.tokenize(target[:-1])

        # input sequence of the decoder - start with sos 
        y_in = torch.tensor(token_seq[:-1])
        # target sequence - ends with eos
        y_out = torch.tensor(token_seq[1:])
        
        return {'melspec': melspec, 'y_in': y_in, 'y_out': y_out}

def create_dataset(text_path, dataset_dir, tokenizer, frontend):
    df = pd.read_csv(text_path, delimiter=' ', header=None, dtype='object')
    df.columns = ['speaker_id', 'chapter_id', 'audio_id', 'seq', 'dataset']

    seq_len = df['seq'].apply(lambda row: len(row)+1)
    return LibriSpeechDataset(df, dataset_dir, tokenizer, frontend), seq_len
    
