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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

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

NUM_VAL_SAMPLES=4
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


def check_for_nan_and_inf(tensor, tensor_name="tensor"):
    if torch.isnan(tensor).any():
        raise ValueError(f"{tensor_name} has NaN values!")
    if torch.isinf(tensor).any():
        raise ValueError(f"{tensor_name} has Inf values!")
    

spectrogram_fn = T.Spectrogram(n_fft=1024, hop_length=256, win_length=1024, power=1.0, normalized=True, center=False).to(device)
amplitude_to_db = T.AmplitudeToDB(stype='power')

def validation(generator, validation_loader, sw, h, steps, device, first=False):
    generator.eval()
    torch.cuda.empty_cache()
    val_err_spec_tot = 0
    val_err_audio_tot = 0

    with torch.no_grad():
        for j, batch in enumerate(validation_loader):
            x_audio, x_spec, y_audio, y_spec, xvector = batch

            try:
                check_for_nan_and_inf(x_audio, "val x_audio")
                check_for_nan_and_inf(x_spec, "val x_spec")
                check_for_nan_and_inf(y_audio, "val y_audio")
                check_for_nan_and_inf(y_spec, "val y_spec")
                check_for_nan_and_inf(xvector, "val xvector")
            except ValueError as e:
                print(e)
                continue

            x_audio, x_spec, y_audio, y_spec, xvector = x_audio.to(device), x_spec.to(device), y_audio.to(device), y_spec.to(device), xvector.to(device)
            y_g_hat = generator(x_audio, x_spec, xvector)
            
            #y_g_hat_spec = spectrogram_fn(y_g_hat.squeeze(1))#.squeeze()

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
            val_err_spec_tot += F.l1_loss(y_spec, y_g_hat_spec).item()
            val_err_audio_tot += F.l1_loss(y_audio, y_g_hat).item()

            if j < NUM_VAL_SAMPLES:
                # Plot spectrograms
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))

                axs[0].imshow(amplitude_to_db(y_spec).squeeze(0).to('cpu').numpy(), origin='lower', aspect='auto')
                axs[0].set_title('Clean Spectrogram')

                axs[1].imshow(amplitude_to_db(y_g_hat_spec).squeeze(0).cpu().numpy(), origin='lower', aspect='auto')
                axs[1].set_title('Denoised Spectrogram')

                axs[2].imshow(amplitude_to_db(x_spec).squeeze(0).cpu().numpy(), origin='lower', aspect='auto')
                axs[2].set_title('Noisy Spectrogram')
                plt.tight_layout()
                sw.add_figure('Spectrograms/Sample_{}'.format(j), fig, steps)
                plt.close(fig)

                sw.add_audio('Audio/Clean_{}'.format(j), y_audio[0], steps, h.sampling_rate)
                sw.add_audio('Audio/Denoised_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)
                sw.add_audio('Audio/Noisy_{}'.format(j), x_audio[0], steps, h.sampling_rate)

        val_spec_err = val_err_spec_tot / (j+1)
        val_audio_err = val_err_audio_tot / (j+1)

        sw.add_scalar("validation/mel_spec_error", val_spec_err, steps)
        sw.add_scalar("validation/audio_error", val_audio_err, steps)

    generator.train()


def train(rank, a, h):
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)
        
    checkpoint_path              = h.checkpoint_path
    checkpoint_cleanunet_path    = h.checkpoint_cleanunet_path
    checkpoint_cleanspecnet_path = h.checkpoint_cleanspecnet_path

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    generator = CleanUNet2(**h.cleanunet2_config).to(device)

    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    # load partial checkpoint models
    if rank == 0:
        if checkpoint_cleanunet_path is not None:
            try:
                print("Loading checkpoint '{}'".format(checkpoint_cleanunet_path))
                generator = load_submodel_checkpoint(generator, checkpoint_cleanunet_path, pre_name="clean_unet.")
            except Exception as e:
                print(e)
            else:
                print("Checkpoint loaded")

        if checkpoint_cleanspecnet_path is not None:
            try:
                print("Loading checkpoint '{}'".format(checkpoint_cleanspecnet_path))
                generator = load_submodel_checkpoint(generator, checkpoint_cleanspecnet_path, pre_name="clean_spec_net.")            
            except Exception as e:
                print(e)
            else:
                print("Checkpoint loaded")


        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'], strict=False)
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if h.freeze_cleanspecnet:
        for param in generator.clean_spec_net.parameters():
            param.requires_grad = False

    if h.freeze_cleanunet:
        for param in generator.clean_unet.parameters():
            param.requires_grad = False
             

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])
    '''
    for param_group in optim_g.param_groups:
        if 'initial_lr' not in param_group:
            param_group['initial_lr'] = param_group['lr']

    for param_group in optim_d.param_groups:
        if 'initial_lr' not in param_group:
            param_group['initial_lr'] = param_group['lr']
    '''
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(h.train_metadata, h.test_metadata)

    trainset = MelDataset(h.data_dir, training_filelist, h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                          shuffle=True if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device,
                          augmentations=h.augmentations)    

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

    if rank == 0:
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
        # first validation when start the train
        #validation(generator, validation_loader, sw, h, steps, device, first=True)

    print_size(generator)

    generator.train()
    mpd.train()
    msd.train()    

    # define multi resolution stft loss    
    if h.loss_config['stft_lambda'] > 0:
        mrstftloss = MultiResolutionSTFTLoss(
                        **h.loss_config['stft_config']
                     ).to(device)
    else:
        mrstftloss = None

    #loss_cleanunet_fn = CleanUNet2Loss(
    #                        **h.loss_config,
    #                        mrstftloss=mrstftloss)
    loss_cleanunet_fn = torch.nn.L1Loss()

    print(f"Epoch {last_epoch + 1}: LR = {scheduler_g.get_last_lr()[0]}")
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            
            x_audio, x_spec, y_audio, y_spec, xvector = batch 
            x_audio, x_spec, y_audio, y_spec, xvector = x_audio.to(device), x_spec.to(device), y_audio.to(device), y_spec.to(device), xvector.to(device)

            # NaN and Inf tensor check
            try:
                check_for_nan_and_inf(x_audio, "x_audio")
                check_for_nan_and_inf(x_spec, "x_spec")
                check_for_nan_and_inf(y_audio, "y_audio")
                check_for_nan_and_inf(y_spec, "y_spec")
                check_for_nan_and_inf(xvector, "xvector")
            except ValueError as e:
                print(e)
                continue

            y_g_hat = generator(x_audio, x_spec, xvector) # (batch_size, 1, segment_size)
            #y_g_hat_spec = spectrogram_fn(y_g_hat).squeeze() # (batch_size, fft, segment_size)
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
            ).squeeze()


            optim_d.zero_grad()

            # MPD
            y_g_hat = y_g_hat.view(y_audio.size(0), 1, -1)
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y_audio, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y_audio, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()

            # Discriminator
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # FRED: Padding mel-spectrograms
            #y_spec, y_g_hat_spec = pad_spectrogram(y_spec, y_g_hat_spec)

            # L1 Mel-Spectrogram Loss
            loss_spec = F.l1_loss(y_spec, y_g_hat_spec) * 45

            #y_audio, y_g_hat = pad_waveform(y_audio, y_g_hat)            
            loss_cleanunet = loss_cleanunet_fn(y_audio, y_g_hat)

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y_audio, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y_audio, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_cleanunet  + loss_spec

            loss_gen_all.backward()

            grad_norm_g = torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=h.grad_clip, norm_type=h.norm_type, error_if_nonfinite=True)
            grad_norm_d = torch.nn.utils.clip_grad_norm_(itertools.chain(msd.parameters(), mpd.parameters()), max_norm=h.grad_clip, norm_type=h.norm_type, error_if_nonfinite=True) 

            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        spec_error = F.l1_loss(y_spec, y_g_hat_spec).item()

                    print('Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                          format(steps, loss_gen_all, spec_error, time.time() - start_b))

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'mpd': (mpd.module if h.num_gpus > 1
                                                         else mpd).state_dict(),
                                     'msd': (msd.module if h.num_gpus > 1
                                                         else msd).state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                     'epoch': epoch})

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/disc_loss_total", loss_disc_all, steps)
                    sw.add_scalar("training/mel_spec_error", spec_error, steps)
                    sw.add_scalar("training/gradient_norm_g", grad_norm_g, steps)
                    sw.add_scalar("training/gradient_norm_d", grad_norm_d, steps)
                    sw.add_scalar("training/lr_gen", optim_g.param_groups[0]["lr"], steps)
                    sw.add_scalar("training/lr_dis", optim_d.param_groups[0]["lr"], steps)

                # Validation
                if steps % a.validation_interval == 0:  # and steps != 0:
                    validation(generator, validation_loader, sw, h, steps, device)
                    generator.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    #parser.add_argument('--input_training_file', default='./filelists/ljs_audio_text_train_filelist.txt')
    #parser.add_argument('--input_validation_file', default='./filelists/ljs_audio_text_val_filelist.txt')
    parser.add_argument('--checkpoint_path', default='logs_training')
    parser.add_argument('--config', default='configs/config.json')
    parser.add_argument('--training_epochs', default=1000000, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
