from .cleanspecnet import CleanSpecNet
from .cleanunet import CleanUNet

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.autograd.set_detect_anomaly(True)

class WaveformConditioner(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(WaveformConditioner, self).__init__()
        self.conv1x1 = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=1
        )  
        self.fn = nn.LeakyReLU(negative_slope=0.4, inplace=False)
    def forward(self, x):
        return self.fn(self.conv1x1(x))
    

# CleanUNet2 (Hybrid Model)
class CleanUNet2(nn.Module):
    def __init__(self, 
            cleanunet_input_channels=1,
            cleanunet_output_channels=1,
            cleanunet_channels_H=64,
            cleanunet_max_H=768,
            cleanunet_encoder_n_layers=8,
            cleanunet_kernel_size=4,
            cleanunet_stride=2,
            cleanunet_tsfm_n_layers=5, 
            cleanunet_tsfm_n_head=8,
            cleanunet_tsfm_d_model=512, 
            cleanunet_tsfm_d_inner=2048,
            cleanspecnet_input_channels=513, 
            cleanspecnet_num_conv_layers=5, 
            cleanspecnet_kernel_size=4, 
            cleanspecnet_stride=1,
            cleanspecnet_num_attention_layers=5, 
            cleanspecnet_num_heads=8, 
            cleanspecnet_hidden_dim=512, 
            cleanspecnet_dropout=0.1):

        super(CleanUNet2, self).__init__()
        
        # Initialize CleanUNet for Waveform Denoising (Waveform-based model)
        self.clean_unet = CleanUNet(
            channels_input=cleanunet_input_channels, 
            channels_output=cleanunet_output_channels,
            channels_H=cleanunet_channels_H, 
            max_H=cleanunet_max_H,
            encoder_n_layers=cleanunet_encoder_n_layers, 
            kernel_size=cleanunet_kernel_size, 
            stride=cleanunet_stride,
            tsfm_n_layers=cleanunet_tsfm_n_layers,
            tsfm_n_head=cleanunet_tsfm_n_head,
            tsfm_d_model=cleanunet_tsfm_d_model, 
            tsfm_d_inner=cleanunet_tsfm_d_inner
        )        

        # Initialize CleanSpecNet for Spectrogram Denoising
        self.clean_spec_net = CleanSpecNet(
            input_channels=cleanspecnet_input_channels, 
            num_conv_layers=cleanspecnet_num_conv_layers, 
            kernel_size=cleanspecnet_kernel_size, 
            stride=cleanspecnet_stride, 
            hidden_dim=cleanspecnet_hidden_dim, 
            num_attention_layers=cleanspecnet_num_attention_layers, 
            num_heads=cleanspecnet_num_heads, 
            dropout=cleanspecnet_dropout
        )
        self.WaveformConditioner = WaveformConditioner()    


    def _reconstruct_waveform(self, noisy_waveform, denoised_spectrogram, n_fft=1024, hop_length=256, window_fn=torch.hann_window):
        # Compute STFT of the noisy waveform to get phase information
        stft_noisy = torch.stft(
            noisy_waveform.squeeze(1),  # Remove channel dimension if necessary
            n_fft=n_fft,
            hop_length=hop_length,
            window=window_fn(n_fft).to(noisy_waveform.device),
            return_complex=True,
            center=False,
            normalized=True
        )  # Shape: (batch_size, freq_bins, time_frames)
        # Get the phase from the noisy STFT
        phase_noisy = torch.angle(stft_noisy)
        # Reconstruct the complex spectrogram using denoised magnitude and noisy phase
        denoised_complex_spectrogram = denoised_spectrogram * torch.exp(1j * phase_noisy)
        # Perform ISTFT to reconstruct waveform
        reconstructed_waveform = torch.istft(
            denoised_complex_spectrogram,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window_fn(n_fft).to(noisy_waveform.device),
            length=noisy_waveform.shape[-1]
        )
        # Add channel dimension back if necessary
        reconstructed_waveform = reconstructed_waveform.unsqueeze(1)
        return reconstructed_waveform
    

    def forward(self, noisy_waveform, noisy_spectrogram):
        denoised_spectrogram = self.clean_spec_net(noisy_spectrogram)
        reconstructed_waveform = self._reconstruct_waveform(noisy_waveform, denoised_spectrogram)
        concat_waveform = torch.cat((noisy_waveform, reconstructed_waveform), dim=1)
        concat_waveform = self.WaveformConditioner(concat_waveform)
        denoised_waveform = self.clean_unet(concat_waveform)
        return denoised_waveform, denoised_spectrogram


# Example usage:
if __name__ == '__main__':

    # Simulated inputs
    noisy_waveform = torch.randn(4, 1, 80000).cuda()  # Waveform input
    clean_waveform = torch.randn(4, 1, 80000).cuda()  # Waveform input
    noisy_spectrogram = torch.randn(4, 513, 309).cuda()  # Spectrogram input
    print("noisy_spectrogram.shape", noisy_spectrogram.shape)

    model = CleanUNet2().cuda()
    print(model)
    print(f"Noisy waveform shape: {noisy_waveform.shape}")
    print(f"Noisy spectrogram shape: {noisy_spectrogram.shape}")
    denoised_waveform, denoised_spec = model(noisy_waveform, noisy_spectrogram)
    print(f"Denoised_waveform waveform shape: {denoised_waveform.shape}")
    print(f"Clean waveform shape: {clean_waveform.shape}")

    loss = torch.nn.MSELoss()(clean_waveform, denoised_waveform)
    loss.backward()
    print(loss.item())

