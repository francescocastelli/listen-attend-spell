import torch
import torchaudio

class MFCCs(torch.nn.Module):
    def __init__(self, sr, n_mfcc, dct_type, norm, n_fft, win_len, hop_len,
                 f_min, f_max, pad, n_mels, window_fn, power, normalized):

        super(MFCCs, self).__init__()
        
        self.pre_emphasis = 0.97
        melkwargs = {'n_fft': n_fft, 'win_length': win_len, 'hop_length': hop_len, 'f_min': f_min,
                     'f_max': f_max, 'pad': pad, 'n_mels': n_mels, 'window_fn': window_fn, 'power': power,
                     'normalized': normalized, 'center': False, 'pad_mode': 'reflect', 'onesided': True,
                     'mel_scale': 'htk'}
        
        self.mfcc = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=n_mfcc, dct_type=dct_type, 
                                               norm='ortho', log_mels=True, melkwargs=melkwargs)
    
    def forward(self, audio):
        audio = torch.squeeze(audio, 1)
        
        mfccs = self.mfcc(audio)
        # mean-normalization
        mfccs -= (torch.mean(mfccs, axis=1) + 1e-8)
        
        return mfccs
