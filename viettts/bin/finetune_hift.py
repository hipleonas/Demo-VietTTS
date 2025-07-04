import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torch.optim as optim
import yaml
import random
import numpy as np
from omegaconf import OmegaConf
"""
https://github.com/kan-bayashi/ParallelWaveGAN
"""
from torch.utils.data import DataLoader, Dataset
# from parallel_wavegan import MultiBandMelGANGenerator
from viettts.hifigan.generator import HiFTGenerator
from viettts.hifigan.f0_predictor import ConvRNNF0Predictor
"""==============This part is for parallel wavegan================"""

def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.

    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.

    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).

    """
    def is_pytorch_17plus():
        version = torch.__version__.split(".")
        major = int(version[0])
        minor = int(version[1])
        return (major > 1) or (major == 1 and minor >= 7)    
    if is_pytorch_17plus:
        x_stft = torch.stft(
            x, fft_size, hop_size, win_length, window, return_complex=False
        )
    else:
        x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-7)).transpose(2, 1)


class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Spectral convergence loss value.

        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Log STFT magnitude loss value.

        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))
    
class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(
        self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"
    ):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()
        # NOTE(kan-bayashi): Use register_buffer to fix #223
        self.register_buffer("window", getattr(torch, window)(win_length))

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).

        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.

        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss

class MultiResolutionSTFTLoss(nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
    ):
        """Initialize Multi resolution STFT loss module.

        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.

        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T) or (B, #subband, T).
            y (Tensor): Groundtruth signal (B, T) or (B, #subband, T).

        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.

        """
        if len(x.shape) == 3:
            x = x.view(-1, x.size(2))  # (B, C, T) -> (B x C, T)
            y = y.view(-1, y.size(2))  # (B, C, T) -> (B x C, T)
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss
"""======================================="""
"""Xử lí dataset"""
class TTSDataset(Dataset):
    def __init__(self, filelist):
        with open(filelist, "r") as f:
            self.items = f.read().splitlines()
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        mel_path, wav_path = self.items[idx].split("|")
        mel = torch.load(mel_path)

        wav , _ = torchaudio.load(wav_path)
        return mel.T, wav[0] #(80,T,(T,)
    
# ----------- Main Training -------------
def main():
    config_path = "/home/hiepquoc/Demo-VietTTS/pretrained-models/hift_config.yaml"
    # with open("config.yaml", "r") as f:
    #     config = yaml.safe_load(f)
    
    config = OmegaConf.load(config_path)
    print('Config loaded from')

    print(config['hift'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    # ---- Instantiate model ----
    f0_predictor = ConvRNNF0Predictor(
        num_class=config["hift"]["f0_predictor"]["num_class"],
        in_channels=config["hift"]["f0_predictor"]["in_channels"],
        cond_channels=config["hift"]["f0_predictor"]["cond_channels"]
    ).to(device)

    hift_config = config["hift"].copy()
    del hift_config["f0_predictor"]
    generator = HiFTGenerator(
        f0_predictor=f0_predictor,  # Truyền riêng
        **hift_config               # Truyền phần còn lại
    ).to(device)


    # ---- Optional: Load pretrained ----
    pretrained_path = config["pretrain_path"]
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained model from: {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location=device)
        generator.load_state_dict(state_dict, strict=False)
    

    # ---- Dataset ----
    # print('FOO')
    # print(config["data"]["path"])
    #Convert to list
    data_path = config["data"]["path"]

    output_list = os.path.join(data_path, "train_filelist.txt")
    wav_files = sorted([f for f in os.listdir(data_path +"/data") if f.lower().endswith(".wav")])
    with open(output_list, "w", encoding="utf-8") as f:
        for wav in wav_files:
            full_path = os.path.join(data_path+"/data", wav)
            f.write(f"{full_path}|{full_path}\n")

    print(f"Đã tạo file list: {output_list} ({len(wav_files)} samples)")

    # dataset = TTSDataset(config["data"]["path"])
    # print(dataset)

    # trainloader = DataLoader(dataset, batch_size = config["batch_size"],shuffle=True, num_workers=4)

    # # ---- Loss & Optimizer ----
    # stft_loss = MultiResolutionSTFTLoss()
    # optimizer = optim.AdamW(generator.parameters(), lr = 1e-4)
    # #---- Training Loop ----
    # max_epochs = config["epochs"]
    # output_dir = config["output_dir"]
    # os.makedirs(output_dir, exist_ok=True)
    # generator.train()
    # step = 0

    # for epoch in range(max_epochs):
    #     for mel, wav in trainloader:
    #         mel = mel.to(device)
    #         wav = wav.to(device)

    #         y_hat, _ = generator(mel)
    #         min_len = min(y_hat.shape[1], wav.shape[1])
    #         y_hat = y_hat[:, :min_len]
    #         wav = wav[:, :min_len]

    #         loss, _ = stft_loss(y_hat, wav)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         if step % 10 == 0:
    #             print(f"Epoch {epoch}, Step {step}: Loss = {loss.item():.4f}")

    #         if step % 1000 == 0:
    #             save_path = os.path.join(output_dir, f"checkpoint_step{step}.pt")
    #             torch.save(generator.state_dict(), save_path)
    #             print(f"Saved checkpoint at: {save_path}")

    #         step += 1
    

if __name__ == "__main__":

   
    main()
