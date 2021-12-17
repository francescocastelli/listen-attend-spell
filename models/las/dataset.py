import torch
import torchaudio
import os
import pandas as pd

class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, df, dataset_dir, tokenizer):
        self.df = df
        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
    
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
        
        target = sample['seq']
        token_seq = self.tokenizer.tokenize(target[:-1])
        
        return {'audio': audio, 'token_seq': token_seq}
