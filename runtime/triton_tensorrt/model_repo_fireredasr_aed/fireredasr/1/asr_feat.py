import math
import os

import numpy as np
import torch
import kaldiio
import kaldifeat

class ASRFeatExtractor:
    def __init__(self, kaldi_cmvn_file, device_id=0):
        self.cmvn = CMVN(kaldi_cmvn_file) if kaldi_cmvn_file != "" else None

        opts = kaldifeat.FbankOptions()
        opts.device = torch.device('cuda', device_id)
        opts.frame_opts.dither = 0
        opts.mel_opts.num_bins = 80
        opts.frame_opts.frame_shift_ms = 10
        opts.frame_opts.frame_length_ms = 25
        opts.frame_opts.samp_freq = 16000
        self.fbank = kaldifeat.Fbank(opts)

    def __call__(self, wavs):
        feats = []
        durs = []
        sample_rate = 16000
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav_tensor = torch.from_numpy(wav)
            else:
                assert isinstance(wav, torch.Tensor), "wav must be a numpy array or a torch tensor"
                wav_tensor = wav
            
            if wav_tensor.ndim == 2:
                wav_tensor = wav_tensor.squeeze(0)

            wav_tensor = wav_tensor.to(torch.float32).contiguous()
            dur = wav_tensor.shape[0] / sample_rate
            fbank = self.fbank(wav_tensor)
            fbank = self.cmvn(fbank)
            feats.append(fbank)
            durs.append(dur)

        lengths = torch.tensor([feat.size(0) for feat in feats])
        feats_pad = self.pad_feat(feats, 0.0)
        return feats_pad, lengths, durs

    def pad_feat(self, xs, pad_value):
        n_batch = len(xs)
        max_len = max([xs[i].size(0) for i in range(n_batch)])
        pad = torch.ones(n_batch, max_len, *xs[0].size()[1:]).to(xs[0].device).to(xs[0].dtype).fill_(pad_value)
        for i in range(n_batch):
            pad[i, :xs[i].size(0)] = xs[i]
        return pad


class CMVN:
    def __init__(self, kaldi_cmvn_file):
        self.dim, self.means, self.inverse_std_variences = \
            self.read_kaldi_cmvn(kaldi_cmvn_file)

    def __call__(self, x, is_train=False):
        assert x.shape[-1] == self.dim, "CMVN dim mismatch"
        out = x - self.means
        out = out * self.inverse_std_variences
        return out

    def read_kaldi_cmvn(self, kaldi_cmvn_file):
        assert os.path.exists(kaldi_cmvn_file)
        stats = kaldiio.load_mat(kaldi_cmvn_file)
        assert stats.shape[0] == 2
        dim = stats.shape[-1] - 1
        count = stats[0, dim]
        assert count >= 1
        floor = 1e-20
        means = []
        inverse_std_variences = []
        for d in range(dim):
            mean = stats[0, d] / count
            means.append(mean.item())
            varience = (stats[1, d] / count) - mean*mean
            if varience < floor:
                varience = floor
            istd = 1.0 / math.sqrt(varience)
            inverse_std_variences.append(istd)
        return dim, np.array(means), np.array(inverse_std_variences)
