import glob
import os
import matplotlib
import torch
from torch.nn.utils import weight_norm
matplotlib.use("Agg")
import matplotlib.pylab as plt


def print_size(net, keyword=None):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, torch.nn.Module):
        # Total number of parameters (trainable and non-trainable)
        total_params = sum(p.numel() for p in net.parameters())
        # Number of trainable parameters
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        # Number of non-trainable parameters
        non_trainable_params = total_params - trainable_params
        
        print("{} - Total Parameters: {:.6f}M; Trainable Parameters: {:.6f}M; Non-trainable Parameters: {:.6f}M".format(
            net.__class__.__name__, total_params / 1e6, trainable_params / 1e6, non_trainable_params / 1e6), flush=True)
        
        if keyword is not None:
            # Parameters associated with the keyword
            keyword_params = [p for name, p in net.named_parameters() if keyword in name]
            keyword_total_params = sum(p.numel() for p in keyword_params)
            keyword_trainable_params = sum(p.numel() for p in keyword_params if p.requires_grad)
            keyword_non_trainable_params = keyword_total_params - keyword_trainable_params
            
            print("'{0}' - Total Parameters: {1:.6f}M; Trainable Parameters: {2:.6f}M; Non-trainable Parameters: {3:.6f}M".format(
                keyword, keyword_total_params / 1e6, keyword_trainable_params / 1e6, keyword_non_trainable_params / 1e6), flush=True)


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def load_submodel_checkpoint(generator, checkpoint_cleanunet_path, pre_name=""):
        if os.path.exists(checkpoint_cleanunet_path):
            checkpoint_dict = torch.load(checkpoint_cleanunet_path, map_location='cpu')    
            new_checkpoint_dict = {}
            for k, v in checkpoint_dict['model_state_dict'].items():
                k = pre_name + k
                new_checkpoint_dict[k] = v
            generator.load_state_dict(new_checkpoint_dict, strict=False)
            return generator        
        else:
            raise FileNotFoundError(f'No valid checkpoint model found at {checkpoint_cleanunet_path}.')            


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

