import torch
import torchaudio

class MelSpectrogram(torch.nn.Module):
    def __init__(self, sr, norm, n_fft, win_len, hop_len,
                 f_min, f_max, pad, n_mels, window_fn, power, normalized):

        super(MelSpectrogram, self).__init__()
        
        self.pre_emphasis = 0.97
        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, win_length=win_len, 
                                             hop_length=hop_len, f_min=f_min, f_max=f_max, pad=pad, 
                                             n_mels=n_mels, window_fn=window_fn, power=power, 
                                             normalized=normalized, center=True, pad_mode='reflect', 
                                             onesided=True, norm=norm, mel_scale='htk')
    
    def forward(self, audio):
        melspec = self.melspec(audio)
        # log melspec
        melspec = torch.log(melspec + 1e-6)
        # mean-normalization
        melspec -= (torch.mean(melspec, axis=0) + 1e-8)
        
        return melspec
