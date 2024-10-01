# Adapted from https://github.com/NVIDIA/waveglow under the BSD 3-Clause License.

# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import os
import argparse
import json
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

from scipy.io.wavfile import write as wavwrite
from scipy.io.wavfile import read as wavread

from util import rescale, find_max_epoch, print_size, sampling
from network import CleanUNet

def denoise(output_dir, ckpt_path, input_dir):
    """
    Denoise audio files in a directory.

    Parameters:
    output_dir (str):               Path to save the denoised audio files.
    ckpt_path (str):                The pretrained checkpoint to be loaded.
    input_dir (str):                Directory containing .wav files to denoise.
    """

    # Setup local experiment path
    exp_path = train_config["exp_path"]
    print('exp_path:', exp_path)

    # Predefine model
    net = CleanUNet(**network_config).cuda()
    print_size(net)

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        os.chmod(output_dir, 0o775)
    print("output dir: ", output_dir, flush=True)

    # Inference
    import glob
    wav_files = glob.glob(os.path.join(input_dir, '*.wav'))
    for wav_file in tqdm(wav_files):
        filename = os.path.basename(wav_file)
        sample_rate, audio = wavread(wav_file)
        assert sample_rate == trainset_config["sample_rate"], "Sample rate mismatch."

        # If audio is stereo, convert to mono
        if len(audio.shape) == 2:
            audio = audio.mean(axis=1)
        # Normalize audio to range [-1, 1]
        if audio.dtype == np.int16:
            max_value = 32768.0  # Maximum value for int16
            audio = audio.astype(np.float32) / max_value
        elif audio.dtype == np.int32:
            max_value = 2147483648.0  # Maximum value for int32
            audio = audio.astype(np.float32) / max_value
        else:
            raise ValueError("Unsupported audio data type: {}".format(audio.dtype))

        audio = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).cuda()  # Shape: [1, 1, L]

        # Denoise audio
        generated_audio = sampling(net, audio)

        # Convert back to original data type
        enhanced_audio = generated_audio[0].squeeze().cpu().numpy()
        enhanced_audio = np.clip(enhanced_audio, -1.0, 1.0)
        output_audio = (enhanced_audio * max_value).astype(np.int16)

        output_file = os.path.join(output_dir, filename)
        wavwrite(output_file, trainset_config["sample_rate"], output_audio)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/DNS-large-full.json',
                        help='JSON file for configuration')
    parser.add_argument('-ckpt_path', '--ckpt_path', default='./exp/DNS-large-full/checkpoint/pretrained.pkl',
                        help='Which checkpoint to use; assign a number or "max" or "pretrained"')     
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help='Directory containing input wav files to denoise')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Directory to save denoised files')
    args = parser.parse_args()

    # Parse configs. Globals are used in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    gen_config      = config["gen_config"]
    global network_config
    network_config  = config["network_config"]      # Network configuration
    global train_config
    train_config    = config["train_config"]        # Training configuration
    global trainset_config
    trainset_config = config["trainset_config"]     # Dataset configurations

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    denoise(output_dir=args.output_dir,
            ckpt_path=args.ckpt_path,
            input_dir=args.input_dir)
