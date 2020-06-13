import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class ExtractAudioFeature(nn.Module):
    def __init__(self, mode="fbank", num_mel_bins=40, **kwargs):
        super(ExtractAudioFeature, self).__init__()
        self.mode = mode
        self.extract_fn = torchaudio.compliance.kaldi.fbank if mode == "fbank" else torchaudio.compliance.kaldi.mfcc
        self.num_mel_bins = num_mel_bins
        self.kwargs = kwargs

    def forward(self, filepath):
        waveform, sample_rate = torchaudio.load(filepath)

        y = self.extract_fn(waveform,
                            num_mel_bins=self.num_mel_bins,
                            channel=-1,
                            sample_frequency=sample_rate,
                            **self.kwargs)
        return y.transpose(0, 1).unsqueeze(0).detach()

    def extra_repr(self):
        return "mode={}, num_mel_bins={}".format(self.mode, self.num_mel_bins)