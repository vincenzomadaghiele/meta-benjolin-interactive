import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
import pickle
import zipfile
from math import ceil


def get_features(audio_signal, frame_length=1024, hop_length=64, device='cpu'):
  """Extract audio features (RMS, zero crossings, spectral flux, spectral flatness).
  
  Args:
      audio_signal: Input audio as torch tensor
      frame_length: Frame length for analysis (default: 1024)
      hop_length: Hop length between frames (default: 64)
      device: Device to use ('cpu' or 'cuda', default: 'cpu')
  
  Returns:
      tuple: (rms_values, zero_crossings, spectral_flux, spectral_flatness)
  """
  num_frames = int(ceil(len(audio_signal) / hop_length))
  
  # Use CPU by default for macOS compatibility
  rms_values = torch.zeros(num_frames, device=device)
  zero_crossings = torch.zeros(num_frames, device=device)
  spectral_flux = torch.zeros(num_frames, device=device)
  spectral_flatness = torch.zeros(num_frames, device=device)
  prev_abs_spectrum = torch.zeros(frame_length // 2 + 1, device=device)
  
  window = torch.hann_window(frame_length, device=device)
  
  for frame_idx in range(num_frames):
    start_index = frame_idx * hop_length
    end_index = min(start_index + frame_length, len(audio_signal))
    frame_data = audio_signal[start_index:end_index]
    if len(frame_data) != len(window):
      window = torch.hann_window(len(frame_data), device=device)
    
    squared_signal = torch.pow(frame_data, 2)
    mean_squared = torch.mean(squared_signal)
    rms_value = torch.sqrt(mean_squared)
    rms_values[frame_idx] = rms_value
    
    sign_diffs = torch.diff(torch.sign(frame_data))
    zero_crossings[frame_idx] = torch.sum(torch.abs(sign_diffs))
    
    abs_spectrum = torch.abs(torch.fft.fft(frame_data))[:frame_length // 2 + 1]
    
    length = min(len(abs_spectrum), len(prev_abs_spectrum))
    spectral_flux[frame_idx] = torch.sum(torch.abs(abs_spectrum[:length] - prev_abs_spectrum[:length]))
    prev_abs_spectrum = abs_spectrum
    
    abs_spectrum[abs_spectrum == 0] += 0.001
    spectral_flatness[frame_idx] = torch.exp(torch.mean(torch.log(abs_spectrum))) / torch.mean(abs_spectrum)

  return rms_values, zero_crossings, spectral_flux, spectral_flatness


class BenjoDataset(Dataset):
    def __init__(self, data_dir, num_mfccs=13, features='mfcc-2d', device='cuda', 
                 fft_args={"win_length": 1024, "hop_size": 64, "pad": 0},
                 weight_normalization=True,
                 feature_dict={"mfcc": True, "centroid": True, "zcr": True, "rms": True, "flux": True, "flatness": True},
                 stat_dictionary=None):
        self.data_dir = data_dir
        self.features = features
        self.device = device
        self.num_mfccs = num_mfccs
        self.win_length = fft_args["win_length"]
        self.hop_size = fft_args["hop_size"]
        self.pad = fft_args["pad"]
        if stat_dictionary:
          self.stat_dictionary = pickle.load(stat_dictionary)
        else:
           self.stat_dictionary = stat_dictionary
        self.feature_dict = feature_dict
        self.weight_normalization = weight_normalization


        self.zip = zipfile.ZipFile(data_dir)
        self.files = self.zip.namelist()[1:]
        # self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".wav")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]

        if self.features == 'params':
            params_array, _ = self.get_benjo_params(index)
            return torch.tensor(params_array, dtype=torch.float32).to(self.device) / 126
        file = self.zip.open(path)
        try:
            waveform, sample_rate = torchaudio.load(file, normalize=True, format='wav')
        except:
            print("Unable to open file")
            return None
        waveform = waveform[0, :].to(self.device)

        if 'bag-of-frames' == self.features:
            feature_list = []
            MFCC = torchaudio.transforms.MFCC(sample_rate=sample_rate,
                                              n_mfcc=self.num_mfccs,
                                              melkwargs={"n_fft": self.win_length,
                                                         "win_length": self.win_length,
                                                         "hop_length": self.hop_size,
                                                         "pad": self.pad,
                                                         "n_mels": 101, "center": False}).to(self.device)
            mfcc = MFCC(waveform)
            length = mfcc.shape[1]

            if self.weight_normalization:
              summary_tensor /= self.num_mfccs
            
            if self.feature_dict["centroid"]:
              get_spectral_centroid = torchaudio.transforms.SpectralCentroid(sample_rate=sample_rate,
                                                               n_fft=self.win_length,
                                                               win_length=self.win_length,
                                                               hop_length=self.hop_size,
                                                               pad=self.pad).to(self.device)
              sp_centroid = get_spectral_centroid(waveform+0.001)[:length]
              
              
            rms, zcr, spectral_flux, spectral_flatness = get_features(waveform, self.win_length, self.hop_size)
            
            
            
            features_tensor = torch.vstack([mfcc, sp_centroid[:length], rms[:length], zcr[:length], spectral_flux[:length], spectral_flatness[:length]])
            
            mean = torch.mean(features_tensor, dim=1, keepdim=True)
            std = torch.std(features_tensor, dim=1, keepdim=True)
            
            summary_tensor = torch.hstack([mean, std])
              
            return summary_tensor

        if 'mfcc' == self.features:
            MFCC = torchaudio.transforms.MFCC(sample_rate=sample_rate,
                                              n_mfcc=self.num_mfccs,
                                              melkwargs={"n_fft": 400,
                                                         "n_mels": 101, "center": False}).to(self.device)
            features = MFCC(waveform)
            features = torch.clip(features, min=-25, max=25)
            shape = features.shape
            features += 25
            features /= 50
            return features[:, :440]

    def get_benjo_params(self, index):
        path = self.files[index]
        params_string = path.removeprefix(self.data_dir + '/').removeprefix('audio/').removesuffix('.wav')
        params_list = params_string.split('-')
        params_array = np.array(list(map(int, params_list)))
        return params_array, params_string

