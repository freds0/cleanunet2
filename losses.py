# Adapted from https://github.com/kan-bayashi/ParallelWaveGAN

# Original Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""STFT-based Loss modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from distutils.version import LooseVersion

is_pytorch_17plus = LooseVersion(torch.__version__) >= LooseVersion("1.7")


def naive_loss_fn(clean_audio, denoised_audio, clean_spec, denoised_spec):
    loss_audio = F.mse_loss(denoised_audio, clean_audio)
    loss_spec = F.l1_loss(denoised_spec, clean_spec)
    loss = loss_audio + loss_spec
    return loss


#def loss_fn(net, X, ell_p, ell_p_lambda, stft_lambda, mrstftloss, **kwargs):
class CleanUnetLoss():
    def __init__(self, ell_p, ell_p_lambda, stft_lambda, mrstftloss, **kwargs):
        """
        Loss function in CleanUNet

        Parameters:
        ell_p: \ell_p norm (1 or 2) of the AE loss
        ell_p_lambda: factor of the AE loss
        stft_lambda: factor of the STFT loss
        mrstftloss: multi-resolution STFT loss function

        """
        self.ell_p = ell_p
        self.ell_p_lambda = ell_p_lambda
        self.stft_lambda = stft_lambda
        self.mrstftloss = mrstftloss

    def __call__(self, clean_audio, denoised_audio):
        """
        Loss Call function
        Parameters:
        clean_audio: clean waveform
        noisy_audio: noisy waveform

        Returns:
        loss: value of objective function
        output_dic: values of each component of loss
        """        
        B, C, L = clean_audio.shape
        output_dic = {}
        loss = 0.0
        
        # AE loss
        #denoised_audio = net(noisy_audio)  

        if self.ell_p == 2:
            ae_loss = nn.MSELoss()(denoised_audio, clean_audio)
        elif self.ell_p == 1:
            ae_loss = F.l1_loss(denoised_audio, clean_audio)
        else:
            raise NotImplementedError
        loss += ae_loss * self.ell_p_lambda
        output_dic["reconstruct"] = ae_loss.data * self.ell_p_lambda

        if self.stft_lambda > 0:
            sc_loss, mag_loss = self.mrstftloss(denoised_audio.squeeze(1), clean_audio.squeeze(1))
            loss += (sc_loss + mag_loss) * self.stft_lambda
            output_dic["stft_sc"] = sc_loss.data * self.stft_lambda
            output_dic["stft_mag"] = mag_loss.data * self.stft_lambda

        return loss, output_dic
    

class CleanUNet2Loss:
    def __init__(self, ell_p, ell_p_lambda, stft_lambda, mrstftloss, **kwargs):
        """
        Initializes the CleanUNet2Loss function with parameters.

        Args:
            ell_p (int): The p value for Lp loss (1 for L1, 2 for L2).
            ell_p_lambda (float): The weight for Lp loss.
            stft_lambda (float): The weight for STFT loss.
            mrstftloss (callable): The multi-resolution STFT loss function.
        """
        self.cleanunet_loss = CleanUnetLoss(ell_p, ell_p_lambda, stft_lambda, mrstftloss)

    def __call__(self, clean_audio, denoised_audio):
        """
        Computes the combined CleanUNet and L1 loss.

        Args:
            clean_audio (Tensor): The clean reference audio.
            denoised_audio (Tensor): The denoised audio output from the model.

        Returns:
            Tensor: The total loss combining CleanUNet and L1 losses.
        """
        # Compute CleanUNet loss
        loss_cleanunet, _ = self.cleanunet_loss(clean_audio, denoised_audio)
        # Compute L1 loss
        loss_l1 = F.l1_loss(clean_audio, denoised_audio, reduction='mean')
        # Return the sum of both losses
        return loss_cleanunet + loss_l1



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
        self, fft_size=1024, shift_size=120, win_length=600, window="hann_window", 
        band="full"
    ):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.band = band 

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

        if self.band == "high":
            freq_mask_ind = x_mag.shape[1] // 2  # only select high frequency bands
            sc_loss  = self.spectral_convergence_loss(x_mag[:,freq_mask_ind:,:], y_mag[:,freq_mask_ind:,:])
            mag_loss = self.log_stft_magnitude_loss(x_mag[:,freq_mask_ind:,:], y_mag[:,freq_mask_ind:,:])
        elif self.band == "full":
            sc_loss  = self.spectral_convergence_loss(x_mag, y_mag)
            mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag) 
        else: 
            raise NotImplementedError

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(
        self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240],
        window="hann_window", sc_lambda=0.1, mag_lambda=0.1, band="full"
    ):
        """Initialize Multi resolution STFT loss module.

        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
            *_lambda (float): a balancing factor across different losses.
            band (str): high-band or full-band loss

        """
        super(MultiResolutionSTFTLoss, self).__init__()
        self.sc_lambda = sc_lambda
        self.mag_lambda = mag_lambda

        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window, band)]

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

        sc_loss *= self.sc_lambda
        sc_loss /= len(self.stft_losses)
        mag_loss *= self.mag_lambda
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss
