"""
Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""

import os
import sys
import time
import math
import random
import logging
import warnings
import argparse
import datetime
import subprocess
from scipy.ndimage import gaussian_filter
from collections import defaultdict, deque

import numpy as np
from PIL import ImageFilter, ImageOps

import torch
import torch.distributed as dist
from torch import nn


def get_shm_size():
    # Execute the 'df -h /dev/shm' command to get the shared memory size in human-readable format
    result = subprocess.run(['df', '-h', '/dev/shm'], stdout=subprocess.PIPE, text=True)
    print("\n", result.stdout)
    return


def num_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return np.round((total_params / 1e6), 3)


def bytes_to_mib(bytes):
    return np.round(bytes / (2**20), 3)


def get_gpu_info(print_info=False):
    
    gpu_info = {}
    for gpu_id in range(torch.cuda.device_count()):
        key = f"GPU_{gpu_id}"
        gpu_info[key] = {
            'id'                : gpu_id,
            'name'              : torch.cuda.get_device_name(gpu_id),
            'memory_allocated'  : bytes_to_mib(torch.cuda.memory_allocated(gpu_id)),
            'memory_reserved'   : bytes_to_mib(torch.cuda.memory_reserved(gpu_id))
        }

    if print_info:
        print(f"\nGPU Information:")
        for gpu_id in list(gpu_info.keys()):
            info = gpu_info[gpu_id]
            print(f"GPU {info['name']} (ID {info['id']}): {info['memory_allocated']} MiB of allocated memory and {info['memory_reserved']} MiB of reserved memory")
    
    return gpu_info


class Memory_Buffer(object):

    def __init__(self):

        self.previous_gpu_info = None

    def __call__(self, operation: str):

        current_gpu_info = get_gpu_info(print_info=False)
        if self.previous_gpu_info is None: # Initialize self.previous_gpu_info
            self.previous_gpu_info = current_gpu_info
        
        print(f"\nGPU memory info after operation: [{operation}]")
        for gpu_id in list(current_gpu_info.keys()):
            new_info, old_info = current_gpu_info[gpu_id], self.previous_gpu_info[gpu_id]
            assert new_info['id'] == old_info['id'], "Error: A rank mismatch occurred during memory logging!"
            assert new_info['name'] == old_info['name'], "Error: A name mismatch occurred during memory logging!"

            new_mem_allocated, new_mem_reserved = new_info['memory_allocated'], new_info['memory_reserved']
            old_mem_allocated, old_mem_reserved = old_info['memory_allocated'], old_info['memory_reserved']
            diff_allocated = np.round(new_mem_allocated - old_mem_allocated, 3)
            diff_reserved = np.round(new_mem_reserved - old_mem_reserved, 3)

            print(f"Device ({new_info['id']}): {new_info['name']}")
            print(f"ABSOLUTE Memory Usage: {new_mem_allocated} MiB allocated and {new_mem_reserved} MiB reserved")
            print(f"DIFFERENCE in Memory Usage: {diff_allocated} MiB allocated and {diff_reserved} MiB reserved")

        self.previous_gpu_info = current_gpu_info # Update step

        return current_gpu_info


def bool_flag(s):

    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}

    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")
    

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print # The original built-in print function is saved to a variable "builtin_print"

    def print(*args, **kwargs): # Define a custom print function
        force = kwargs.pop('force', False) # Extract "force" keyword from kwargs (with a default value of "False" if not provided)
        # If process is the master process or if "force" is True,
        # indicating that printing should occur regardless of whether the process is the master,
        # print the output.
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print # Assign custom print function to __builtin__.print, effectively overriding the built-in print
    

def init_distributed_mode(args):

    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])

    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()

    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    # Initialize the default distributed process group.
    # This will also initialize the distributed package.
    dist.init_process_group(
        backend="nccl", # valid values include mpi, gloo, nccl, and ucc
        init_method=args.dist_url, # URL specifying how to initialize the process group. Default is “env://” meaning to use environment variables for initialization.
        world_size=args.world_size, # number of processes participating in the job
        rank=args.rank, # rank of the current process (it should be a number between 0 and world_size-1)
    )

    torch.cuda.set_device(args.gpu) # set your device to local rank
    print("-> Distributed Mode Inititialization (rank {}): '{}'".format(args.rank, args.dist_url), flush=True)
    dist.barrier() # synchronize all processes
    setup_for_distributed(args.rank == 0)


def fix_random_seeds(seed=31): # Fix random seeds for generating random numbers on all devices
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_sha(): # Get Secure Hash Algorithm (SHA)
    """
    SHA, or Secure Hash Algorithm, is an algorithm used in git to uniquely identify commits.
    SHA produces a fixed-length hash (digest) from a variable-length input.
    """
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'

    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass

    message = f"SHA: {sha}, Status: {diff}, Branch: {branch}"

    return message


class GaussianBlur(object): # Apply Gaussian Blur to PIL image

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):

        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max))
            # radius – Standard deviation of the Gaussian kernel.
            # Either a sequence of two numbers for x and y, or a single number for both.
        )


class Solarization(object): # Apply Solarization to PIL image
    '''
    Invert all pixel values above a threshold, threshold simply means image segmentation.
    Default: PIL.ImageOps.solarize(image: Image, threshold: int = 128)
    '''

    def __init__(self, p, threshold=128):
        self.p = p
        self.threshold = threshold # additionally added

    def __call__(self, img):

        if random.random() < self.p:
            return ImageOps.solarize(img, threshold=self.threshold)
        else:
            return img
        

##############################################
""" Version 1 """
##############################################
def preprocess_image(_img, clahe=False, clahe_clip_limit=3.0, clahe_window=(8,8), subtract_lowpass=True, dtype='float32'):

    if dtype == 'float16':
        dtype = np.float16
    elif dtype == 'float32':
        dtype = np.float32
    elif dtype == 'float64':
        dtype = np.float64
    else:
        raise ValueError("Please specify a valid datatype (either float16, float32, or float64) for casting ...")

    # asserting assumptions
    assert (_img.ndim == 2) and (_img.dtype == np.uint16)
    _img = _img.astype(dtype, casting='safe')  # cast to higher bit-depth to avoid data loss during rescaling
    raw_image = _img

    # preprocess only inner region (1/4 image margin) (avoids pixel-errors at boarder)
    margin = _img.shape[0] // 8
    ROI = _img[margin:-margin, margin:-margin]
    cval = np.median(ROI) + 3 * np.std(ROI)  # set contrast window upper limit to median + 3*std
    if cval == 0:
        logging.warning("cval is zero... this is likely a mistake?")
        return np.zeros_like(_img, dtype=dtype)
    _img = np.minimum(_img, cval) / cval  # clip and normalize to 1
    assert _img.dtype == dtype

    # apply neglog
    _img = -np.log(np.maximum(_img, np.finfo(dtype=_img.dtype).eps))
    assert _img.dtype == dtype
    
    # apply histogram equalization
    if clahe:
        raise ValueError("Error: CLAHE not supported yet!")
        # assert _img.dtype == np.uint8
        # clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_window)
        # _img = clahe.apply(_img)

    # calculate lowpass component and subtract to mitigate intensity gradients
    if subtract_lowpass:
        if _img.shape != (976, 976):
            logging.warning("Gaussian size is adapted to standard image size... You might want to check this!")
        lowpass = gaussian_filter(_img, sigma=20/0.305) # sigma ~ 20mm (Standard deviation for the Gaussian kernel) 
        _img -= lowpass
        assert _img.dtype == dtype

    # scale contrast to [0.0, 1.0]
    ROI = _img[margin:-margin, margin:-margin]
    _img = (_img - ROI.min()) / (ROI.max() - ROI.min())
    assert _img.dtype == dtype

    '''
    The following clipping operation was additionally added and not contained in the original implementation!
    It is applied to ensure a consistent value range of [0.0, 1.0] for the preprocessed images.
    Since the histogram is mainly distributed on the interval [0.0, 1.0], this doesn't change the image characteristics much ...
    '''
    _img = np.clip(a=_img, a_min=0.0, a_max=1.0) # Additionally added

    return _img, (raw_image / 65535.0).astype(dtype, casting='safe')


##############################################
""" Version 2 """
# Random I0-Normalization and Neglog-Transform
##############################################
def preprocess_image_v2(_img, clahe=False, clahe_clip_limit=3.0, clahe_window=(8,8), subtract_lowpass=True, dtype='float32'):

    if dtype == 'float16':
        dtype = np.float16
    elif dtype == 'float32':
        dtype = np.float32
    elif dtype == 'float64':
        dtype = np.float64
    else:
        raise ValueError("Please specify a valid datatype (either float16, float32, or float64) for casting ...")

    # asserting assumptions
    assert (_img.ndim == 2) and (_img.dtype == np.uint16)
    _img = _img.astype(dtype, casting='safe') # cast to higher bit-depth to avoid data loss during rescaling
    raw_image = _img

    #########################################################################
    """ Randomly apply I0-Normalization and Neglog-Transform with p = 0.5 """
    #########################################################################
    margin = _img.shape[0] // 8
    if random.random() < 0.5:

        # preprocess only inner region (1/4 image margin) (avoids pixel-errors at boarder)
        ROI = _img[margin:-margin, margin:-margin]
        ###############################################################
        """ Randomly choose contrast scaling factor from [2.0, 3.5] """
        contrast_factor = random.uniform(2.0, 3.5)
        """ Use a fixed contrast scaling factor of 3.0 """
        # contrast_factor = 3.0
        # print(f"Contrast factor = {contrast_factor}")
        ###############################################################
        cval = np.median(ROI) + contrast_factor * np.std(ROI) # set contrast window upper limit to median + 3*std
        if cval == 0:
            logging.warning("cval is zero... this is likely a mistake?")
            return np.zeros_like(_img, dtype=dtype)
        _img = np.minimum(_img, cval) / cval # clip and normalize to 1
        assert _img.dtype == dtype

        # apply neglog
        _img = -np.log(np.maximum(_img, np.finfo(dtype=_img.dtype).eps))
        assert _img.dtype == dtype
    
        # apply histogram equalization
        if clahe:
            raise ValueError("Error: CLAHE not supported yet!")
            # assert _img.dtype == np.uint8
            # clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_window)
            # _img = clahe.apply(_img)

        # calculate lowpass component and subtract to mitigate intensity gradients
        if subtract_lowpass:
            if _img.shape != (976, 976):
                logging.warning("Gaussian size is adapted to standard image size... You might want to check this!")
            lowpass = gaussian_filter(_img, sigma=20/0.305) # sigma ~ 20mm (Standard deviation for the Gaussian kernel) 
            _img -= lowpass
            assert _img.dtype == dtype

    # scale contrast to [0.0, 1.0]
    ROI = _img[margin:-margin, margin:-margin]
    _img = (_img - ROI.min()) / (ROI.max() - ROI.min())
    assert _img.dtype == dtype

    '''
    The following clipping operation was additionally added and not contained in the original implementation!
    It is applied to ensure a consistent value range of [0.0, 1.0] for the preprocessed images.
    Since the histogram is mainly distributed on the interval [0.0, 1.0], this doesn't change the image characteristics much ...
    '''
    _img = np.clip(a=_img, a_min=0.0, a_max=1.0) # Additionally added

    return _img, (raw_image / 65535.0).astype(dtype, casting='safe')


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """
    Cut & paste from PyTorch official master until it's in a few official releases - RW
    Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)

        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """ type: (Tensor, float, float, float, float) -> Tensor """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()

        # Disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):

        # Convert to list
        if not isinstance(x, list):
            x = [x]

        idx_crops = torch.cumsum(
            input = torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True, # Whether to also return the counts for each unique element.
                )[1], # Returns (output, counts) = (unique scalar elements, number of occurrences for each unique value)
            dim = 0) # Returns cumulative sum of elements of input in the specified dimension
        
        """
        Example for torch.cumsum():

        >>> a = torch.randint(1, 20, (10,))
        >>> a
        tensor([13,  7,  3, 10, 13,  3, 15, 10,  9, 10])
        >>> torch.cumsum(a, dim=0)
        tensor([13, 20, 23, 33, 46, 49, 64, 74, 83, 93])
        """

        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:

            _out = self.backbone(torch.cat(x[start_idx : end_idx]))

            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]

            # Accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx

        # Run the head forward on the concatenated features
        return self.head(output)


def has_batchnorms(model): 
    """
    Ceck whether model contains any batch norm layers
    """
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def get_params_groups(model):
    """
    Determine model parameters, which should be regularized or not
    """
    regularized = []
    not_regularized = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # We do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)

    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.0}]


class LARS(torch.optim.Optimizer): # to use with convnet and large batches
    """
    Almost copy-paste from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim != 1:
                    dp = dp.add(p, alpha=g['weight_decay'])

                if p.ndim != 1:
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    """
    Set up a cosine scheduler instance
    """
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep

    return schedule


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):

    if not os.path.isfile(ckp_path):
        print("\nCould not find any checkpoint at '{}'".format(ckp_path))
        print("Starting a complete new training run ...")
        return
    
    else:
        print("\nFound checkpoint at '{}'".format(ckp_path))

        checkpoint = torch.load(ckp_path, map_location="cpu", weights_only=False)

        for key, value in kwargs.items():
            if key in checkpoint and value is not None:
                try:
                    msg = value.load_state_dict(checkpoint[key], strict=True) # Original code: strict=False
                    print("=> loaded '{}' from checkpoint '{}' with msg '{}'".format(key, ckp_path, msg))
                except TypeError:
                    try:
                        msg = value.load_state_dict(checkpoint[key])
                        print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                    except ValueError:
                        print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
            else:
                print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

        # Reload variables important for the run (e.g., "epoch")
        if run_variables is not None:
            for var_name in run_variables:
                if var_name in checkpoint:
                    run_variables[var_name] = checkpoint[var_name]

        print(f"\nSuccessfully restored previous training up to epoch {run_variables['epoch']}") # Additionally added

        return


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank() # Return the rank of the current process in the provided group ranging from 0 to world_size


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


class SmoothedValue(object):
    """
    Track a series of values and provide access to smoothed values over a window or the global series average.
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            # fmt = "{median:.6f} ({global_avg:.6f})" # Original code
            fmt = "{global_avg:.6f}" # -> global_avg is also used for saving logs in log.txt
        self.fmt = fmt
        self.deque = deque(maxlen=window_size)
        self.count = 0
        self.total = 0.0

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: Does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')

        dist.barrier()
        # Synchronize all processes

        dist.all_reduce(t)
        # Reduces the tensor data across all machines in a way that all get the final result.
        # After the call the tensor is going to be bitwise identical in all processes.

        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median = self.median,
            avg = self.avg,
            global_avg = self.global_avg,
            max = self.max,
            value = self.value)


class MetricLogger(object):

    def __init__(self, delimiter="\t"):
        
        self.meters = defaultdict(SmoothedValue)
        # Note: defaultdict never raises a KeyError! It provides a default value for the key that does not exist.
        # -> This 'default value' is determined by the function or object passed to defaultdict(), i.e., SmoothedValue

        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):

        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'

        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.1f} MB'])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'])

        MB = 1024.0 * 1024.0
        for obj in iterable:

            data_time.update(time.time() - end)

            yield obj
            """
            The yield keyword is used to return a list of values from a function.
            Unlike the return keyword which stops further execution of the function,
            the yield keyword continues to the end of the function. When you call a function
            with yield keyword(s), the return value will be a list of values, one for each yield.
            """

            iter_time.update(time.time() - end)

            # if (i % print_freq) == 0 or (i == len(iterable)) - 1: # Original code
            if ((i % print_freq) == 0) or (i == (len(iterable) - 1)): # Adapted code

                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                if torch.cuda.is_available():
                    print(log_msg.format(i+1, len(iterable),
                                         eta = eta_string,
                                         meters = str(self),
                                         time = str(iter_time),
                                         data = str(data_time),
                                         memory = torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(i+1, len(iterable),
                                         eta = eta_string,
                                         meters = str(self),
                                         time = str(iter_time),
                                         data = str(data_time)))
                    
            i += 1
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(('{}' + self.delimiter +  'Total Time: {} (-> {:.3f} seconds / batch)').format(header, total_time_str, total_time / len(iterable)))


def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):

    if os.path.isfile(pretrained_weights):

        # print(f"\nModel parameters: {list(model.state_dict().keys())}") # Additionally added

        state_dict = torch.load(pretrained_weights, map_location="cpu", weights_only=False)
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"\nTake key [{checkpoint_key}] in provided checkpoint dict ...")
            state_dict = state_dict[checkpoint_key]

        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # remove `backbone.` prefix induced by multicrop wrapper
        # state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()} # Deactivated due to compatibility issue

        # msg = model.load_state_dict(state_dict, strict=False) # Original code
        msg = model.load_state_dict(state_dict, strict=True) # Adapted code
        # If strict is True, then the keys of state_dict must exactly match the keys returned by this module’s state_dict() function
        print('Pretrained weights found at {} and loaded with message: {}'.format(pretrained_weights, msg))

    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate!")
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        elif model_name == "xcit_small_12_p16":
            url = "dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth"
        elif model_name == "xcit_small_12_p8":
            url = "dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth"
        elif model_name == "xcit_medium_24_p16":
            url = "dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth"
        elif model_name == "xcit_medium_24_p8":
            url = "dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth"
        elif model_name == "resnet50":
            url = "dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights ...")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
            # If strict is True, then the keys of state_dict must exactly match the keys returned by this module’s state_dict() function
        else:
            print("There is no reference weights available for this model! => We use random weights ...")


def load_pretrained_linear_weights(linear_classifier, model_name, patch_size):
    url = None
    if model_name == "vit_small" and patch_size == 16:
        url = "dino_deitsmall16_pretrain/dino_deitsmall16_linearweights.pth"
    elif model_name == "vit_small" and patch_size == 8:
        url = "dino_deitsmall8_pretrain/dino_deitsmall8_linearweights.pth"
    elif model_name == "vit_base" and patch_size == 16:
        url = "dino_vitbase16_pretrain/dino_vitbase16_linearweights.pth"
    elif model_name == "vit_base" and patch_size == 8:
        url = "dino_vitbase8_pretrain/dino_vitbase8_linearweights.pth"
    elif model_name == "resnet50":
        url = "dino_resnet50_pretrain/dino_resnet50_linearweights.pth"
    if url is not None:
        print("We load the reference pretrained linear weights.")
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)["state_dict"]
        linear_classifier.load_state_dict(state_dict, strict=True)
    else:
        print("We use random linear weights.")


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # Sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class PCA():
    """
    Class to  compute and apply PCA.
    """
    def __init__(self, dim=256, whit=0.5):

        self.dim = dim
        self.whit = whit
        self.mean = None

    def train_pca(self, cov): # Takes a covariance matrix (np.ndarray) as input
        d, v = np.linalg.eigh(cov)
        eps = d.max() * 1e-5
        n_0 = (d < eps).sum()
        if n_0 > 0:
            d[d < eps] = eps

        # Total energy
        totenergy = d.sum()

        # Sort eigenvectors with eigenvalues order
        idx = np.argsort(d)[::-1][:self.dim]
        d = d[idx]
        v = v[:, idx]

        print("keeping %.2f %% of the energy" % (d.sum() / totenergy * 100.0))

        # For the whitening
        d = np.diag(1. / d**self.whit)

        # Principal components
        self.dvt = np.dot(d, v.T)

    def apply(self, x):

        # Input is from numpy
        if isinstance(x, np.ndarray):
            if self.mean is not None:
                x -= self.mean
            return np.dot(self.dvt, x.T).T

        # Input is from torch and is on GPU
        if x.is_cuda:
            if self.mean is not None:
                x -= torch.cuda.FloatTensor(self.mean)
            return torch.mm(torch.cuda.FloatTensor(self.dvt), x.transpose(0, 1)).transpose(0, 1)

        # Input if from torch, on CPU
        if self.mean is not None:
            x -= torch.FloatTensor(self.mean)

        return torch.mm(torch.FloatTensor(self.dvt), x.transpose(0, 1)).transpose(0, 1)


def compute_ap(ranks, nres):

    """
    Computes average precision for given ranked indexes.
        ranks : zerro-based ranks of positive images
        nres  : number of positive images
    """

    # Number of images ranked by the system
    nimgranks = len(ranks)

    # Accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap


def compute_map(ranks, gnd, kappas=[]):

    """
    Computes the mAP for a given set of returned results.
         Usage:
           map = compute_map (ranks, gnd)
                 computes mean average precsion (map) only
           map, aps, pr, prs = compute_map (ranks, gnd, kappas)
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query
         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.
    nq = len(gnd) # Number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # No positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # Sorted positions of positive and junk images (0 based)
        pos  = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgndj)]

        k = 0;
        ij = 0;
        if len(junk):
            # Decrease positions of positives based on the number of
            # Junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # Compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # Compute precision @ k
        pos += 1 # Get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j]); 
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)

    return map, aps, pr, prs


def multi_scale(samples, model):
    v = None
    for s in [1, 1/2**(1/2), 1/2]: # We use 3 different scales
        if s == 1:
            inp = samples.clone()
        else:
            inp = nn.functional.interpolate(samples, scale_factor=s, mode='bilinear', align_corners=False)
        feats = model(inp).clone()
        if v is None:
            v = feats
        else:
            v += feats
    v /= 3
    v /= v.norm()
    return v
