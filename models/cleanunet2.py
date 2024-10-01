from .cleanspecnet import CleanSpecNet
from .cleanunet import CleanUNet

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

#torch.autograd.set_detect_anomaly(True)

class SpectrogramUpSampler(nn.Module):
    """
    Rede neural para up-sampling de espectrogramas usando convoluções transpostas 2D.
    
    Recebe um espectrograma com dimensões (Batch_size, n_freq=513, len_seq)
    e realiza up-sampling por um fator de ~304 na dimensão temporal através de
    duas camadas de convolução transposta 2D.
    """
    def __init__(self, n_freq=513, upsample_factor=304, alpha=0.4):
        super(SpectrogramUpSampler, self).__init__()
        
        self.upsample_factor = upsample_factor
        self.alpha = alpha
        
        # Primeira camada de convolução transposta com fator de ~17
        self.conv_trans1 = nn.ConvTranspose2d(
            in_channels=1,
            out_channels=16,
            kernel_size=(33, 4),
            stride=(1, 16),  # Fator de upsampling 16x
            padding=(16, 1),
            bias=False
        )
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=self.alpha, inplace=False)
        
        # Segunda camada de convolução transposta com fator de ~16 para chegar no fator total ~256
        self.conv_trans2 = nn.ConvTranspose2d(
            in_channels=16,
            out_channels=1,
            kernel_size=(33, 4),
            stride=(1, 16),  # Fator de upsampling 16x
            padding=(16, 1),
            bias=False
        )
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=self.alpha, inplace=False)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_trans1(x)
        x = self.leaky_relu1(x)
        x = self.conv_trans2(x)
        x = self.leaky_relu2(x)
        x = x.squeeze(1)
        return x


class SpectrogramReducer(nn.Module):
    def __init__(self, in_channels=513, out_channels=1):
        super(SpectrogramReducer, self).__init__()
        self.conv1x1 = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=1
        )
        
    def forward(self, x):
        return self.conv1x1(x)


class CleanSpecNetPosNet(nn.Module):
    def __init__(self, in_channels=513, out_channels=1):
        super(CleanSpecNetPosNet, self).__init__()
        self.upsampler = SpectrogramUpSampler()
        self.reducer = SpectrogramReducer()

    def forward(self, x):
        upsampled_spectrogram = self.upsampler(x)
        reduced_spectrogram = self.reducer(upsampled_spectrogram)
        return reduced_spectrogram


class CleanUNetPreNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(CleanUNetPreNet, self).__init__()
        self.conv1x1 = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=1
        )
        
    def forward(self, x):
        return self.conv1x1(x)
 
            
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
        self.cleanspecnet_posnet = CleanSpecNetPosNet()
        self.cleanunet_prenet = CleanUNetPreNet()


    def forward(self, noisy_waveform, noisy_spectrogram):

        denoised_spectrogram = self.clean_spec_net(noisy_spectrogram)
        reduced_spectrogram = self.cleanspecnet_posnet(denoised_spectrogram)

        if reduced_spectrogram.shape[-1] < noisy_waveform.shape[-1]:
            padded_reduced_spectrogram = F.pad(reduced_spectrogram, (0, noisy_waveform.shape[-1] - reduced_spectrogram.shape[-1]))
            padded_noisy_waveform = noisy_waveform
        else:
            padded_reduced_spectrogram = reduced_spectrogram
            padded_noisy_waveform = F.pad(noisy_waveform, (0, reduced_spectrogram.shape[-1] - noisy_waveform.shape[-1]))

        conditioned_waveform = torch.cat((padded_noisy_waveform, padded_reduced_spectrogram), dim=1)

        conditioned_waveform = self.cleanunet_prenet(conditioned_waveform)
                
        denoised_waveform = self.clean_unet(conditioned_waveform)
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
    print(f"denoised_waveform waveform shape: {denoised_waveform.shape}")
    print(f"Clean waveform shape: {clean_waveform.shape}")

    loss = torch.nn.MSELoss()(clean_waveform, denoised_waveform)
    loss.backward()
    print(loss.item())

