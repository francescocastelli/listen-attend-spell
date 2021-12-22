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
        # ensure a melspec with even length 
        if melspec.shape[0] % 2:
            melspec = torch.nn.functional.pad(melspec, (0, 0, 0, 1))
        
        # tokenize the target seq
        target = sample['seq']
        token_seq = torch.tensor(self.tokenizer.tokenize(target[:-1]))
        
        return {'melspec': melspec, 'token_seq': token_seq}

def create_dataset(text_path, dataset_dir, tokenizer, frontend):
    df = pd.read_csv(text_path, delimiter=' ', header=None, dtype='object')
    df.columns = ['speaker_id', 'chapter_id', 'audio_id', 'seq', 'dataset']

    return LibriSpeechDataset(df, dataset_dir, tokenizer, frontend)
    
