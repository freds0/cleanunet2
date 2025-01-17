import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
import torchaudio.transforms as T
import torchaudio.functional as F_audio
from env import AttrDict, build_env
#from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from spec_dataset import MelDataset, mel_spectrogram, get_dataset_filelist
from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint, print_size, load_submodel_checkpoint

from cleanunet import CleanUNet2
from losses import MultiResolutionSTFTLoss, CleanUnetLoss, CleanUNet2Loss

import warnings
warnings.filterwarnings("ignore", message=".*had to be resampled.*")

torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on device: {}".format(device))

N_FFT=1024
HOP_LENGTH=256 
WIN_LENGTH=1024
POWER=1.0
NORMALIZED=True
CENTER=False

'''
def pad_spectrogram(spec1, spec2):
    # Source: https://github.com/jik876/hifi-gan/issues/52
    if spec1.size(2) > spec2.size(2):
        spec2 = torch.nn.functional.pad(spec2, (0, spec1.size(2) - spec2.size(2)), 'constant')
    elif spec1.size(2) < spec2.size(2):
        spec1 = torch.nn.functional.pad(spec1, (0, spec2.size(2) - spec1.size(2)), 'constant')
    return spec1, spec2
'''
def pad_spectrogram(spec1, spec2):
    # Ensure tensors have 3 dimensions
    if len(spec1.size()) < 3 or len(spec2.size()) < 3:
        raise ValueError("Expected spectrograms with at least 3 dimensions, but received: spec1.size={} and spec2.size={}".format(spec1.size(), spec2.size()))

    # Source: https://github.com/jik876/hifi-gan/issues/52
    if spec1.size(2) > spec2.size(2):
        spec2 = torch.nn.functional.pad(spec2, (0, spec1.size(2) - spec2.size(2)), 'constant')
    elif spec1.size(2) < spec2.size(2):
        spec1 = torch.nn.functional.pad(spec1, (0, spec2.size(2) - spec1.size(2)), 'constant')
    return spec1, spec2


def pad_waveform(wav1, wav2):
    if wav1.size(2) > wav2.size(2):
        wav2 = torch.nn.functional.pad(wav2, (0, wav1.size(2) - wav2.size(2)), 'constant')
    elif wav1.size(2) < wav2.size(2):
        wav1 = torch.nn.functional.pad(wav1, (0, wav2.size(2) - wav1.size(2)), 'constant')
    return wav1, wav2


#spectrogram_fn = T.Spectrogram(n_fft=1024, hop_length=256, win_length=1024, power=1.0, normalized=True, center=False).to(device)

def validation(generator, validation_loader, sw, h, steps, device, first=False):
    generator.eval()
    torch.cuda.empty_cache()
    val_err_tot = 0
    with torch.no_grad():
        for j, batch in enumerate(validation_loader):
            x_audio, x_spec, y_audio, y_spec, xvector = batch
            x_audio, x_spec, y_audio, y_spec, xvector = x_audio.to(device), x_spec.to(device), y_audio.to(device), y_spec.to(device), xvector.to(device)
            y_g_hat = generator(x_audio, x_spec, xvector)
            
            #val_err_tot += F.l1_loss(y_spec, y_g_hat_spec).item()
            val_err = F.l1_loss(y_audio, y_g_hat)
            print(val_err.item())
            val_err_tot += val_err

            #y_g_hat_spec = spectrogram_fn(y_g_hat.squeeze(1))#.squeeze()
            '''
            window = torch.hann_window(N_FFT).to(y_g_hat.get_device())
            y_g_hat_spec = F_audio.spectrogram(
                y_g_hat.squeeze(1),
                pad=0,
                window=window,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                win_length=WIN_LENGTH,
                power=POWER,  # Use 2.0 for power spectrogram or None for complex spectrogram
                normalized=NORMALIZED,
                center=CENTER
            )
            # FRED: upsampling
            y_spec, y_g_hat_spec = pad_spectrogram(y_spec, y_g_hat_spec)
            val_err_tot += F.l1_loss(y_spec, y_g_hat_spec).item()

            if j <= 4:
                if steps == 0 or first:
                    y_spec = mel_spectrogram(y_audio[0].squeeze(1), h.n_fft, h.num_mels,
                                                    h.sampling_rate, h.hop_size, h.win_size,
                                                    h.fmin, h.fmax)                    
                    sw.add_audio('target/y_{}'.format(j), y_audio[0], steps, h.sampling_rate)
                    sw.add_figure('target/y_spec_{}'.format(j), plot_spectrogram(y_spec.squeeze(0).to('cpu')), steps)

                sw.add_audio('denoised/y_hat_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)
                y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                                h.sampling_rate, h.hop_size, h.win_size,
                                                h.fmin, h.fmax)
                sw.add_figure('denoised/y_hat_spec_{}'.format(j),
                                plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)


                sw.add_audio('noisy/x_{}'.format(j), x_audio[0], steps, h.sampling_rate)
                x_hat_spec = mel_spectrogram(x_audio.squeeze(1), h.n_fft, h.num_mels,
                                                h.sampling_rate, h.hop_size, h.win_size,
                                                h.fmin, h.fmax)
                sw.add_figure('noisy/x_spec_{}'.format(j),
                                plot_spectrogram(x_hat_spec.squeeze(0).cpu().numpy()), steps)
            '''
        val_err = val_err_tot / (j+1)

        print("validation/mel_spec_error: ", val_err.item())


def run_test(rank, a, h):

    checkpoint_path              = h.checkpoint_path

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    generator = CleanUNet2(**h.cleanunet2_config).to(device)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    if cp_g is None or cp_do is None:
        print("Error: No checkpoint found")
        return
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        generator.load_state_dict(state_dict_g['generator'])

    training_filelist, validation_filelist = get_dataset_filelist(h.train_metadata, h.test_metadata)

    validset = MelDataset(h.data_dir, validation_filelist, h.segment_size, h.n_fft, h.num_mels,
                            h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, n_cache_reuse=0,
                            fmax_loss=h.fmax_for_loss, device=device,
                            augmentations=h.augmentations)
    
    validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                    sampler=None,
                                    batch_size=1,
                                    pin_memory=True,
                                    drop_last=True)

    sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    validation(generator, validation_loader, sw, h, 0, device)


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default='logs_training')
    parser.add_argument('--config', default='configs/config.json')


    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
    else:
        pass

    run_test(0, a, h)


if __name__ == '__main__':
    main()
