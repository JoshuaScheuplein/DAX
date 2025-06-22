import os
import sys
import json
import math
import time
import random
import platform
import datetime
import argparse
from pathlib import Path

import wandb
from azureml.core import Run

import numpy as np
# from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision import models as torchvision_models
from torchvision.transforms import InterpolationMode

import utils
#####################################
# import vision_transformer as vits
import vision_transformer_dax as vits
#####################################
from vision_transformer import DINOHead
from datasets import AzureDataset, DummyDataset
from datasets import AzureDatasetV2, DummyDatasetV2


torchvision_archs = sorted(name for name in torchvision_models.__dict__
                    if name.islower() and not name.startswith("__")
                    and callable(torchvision_models.__dict__[name]))


"""
'RANK' -> The RANK environment variable typically refers to the global rank of a process in a distributed training setup.
The global rank is a unique identifier for each process across all nodes and GPUs. For example, if you are training
on 2 nodes with 4 GPUs each (total 8 GPUs), the global rank will range from 0 to 7.

'LOCAL_RANK' -> The LOCAL_RANK environment variable refers to the rank of a process on a specific node.
In a multi-node setup, each node has its own set of GPUs, and the local rank differentiates processes within a single node.
For example, on a node with 4 GPUs, the local rank will range from 0 to 3.
"""


""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
"""
We set <max_split_size_mb> in order to avoid memory fragmentation and any cuda realted OOM error 
(torch.cuda.OutOfMemoryError).
"""
if not platform.system() == "Windows":
    max_mb = 32 # NOTE: max_split_size_mb must be > 20
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"backend:native, max_split_size_mb:{max_mb}"

    # If-condition is needed since setup for torch.distributed has not yet been called!
    if int(os.environ["RANK"]) == 0: # Check the global rank
        print("\nPYTORCH_CUDA_ALLOC_CONF:", os.environ["PYTORCH_CUDA_ALLOC_CONF"], "\n")
""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """


def get_args_parser():

    parser = argparse.ArgumentParser('DAX', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        # choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
        #         + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"), # Original code
        choices=['resnet18', 'resnet50', 'vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'], # Adapted code
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--azure', type=utils.bool_flag, default=True, help='Whether training happens on Azure or local cluster.')
    parser.add_argument('--dataset', default='', type=str, help='Name of the used training dataset.')
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str, help='Please specify path to the training dataset.')
    parser.add_argument('--data_csv_path', default='', type=str, help='Please specify CSV file path containing dataset info.')
    parser.add_argument('--data_mean', type=float, default=None, help='Please specify mean value of training dataset.')
    parser.add_argument('--data_std', type=float, default=None, help='Please specify std value of training dataset.')
    parser.add_argument('--augmentation', default='v1', type=str, help='Version of data augmentation strategy.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--use_wandb', type=utils.bool_flag, default=True, help='Whether to use W&B library for logging or not.')
    parser.add_argument('--subtract_lowpass', type=utils.bool_flag, default=False, help='Subtract lowpass signal from input imagess.')
    parser.add_argument('--saveckp_freq', default=50, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--job_ID', default="Test-Job", type=str, help='Job descriptor or ID')
    parser.add_argument('--pretrained_weights', type=str, default='No Pretraining', help='Whether to load pretrained weights or not.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""URL used to set up distributed training;
                        see https://pytorch.org/docs/stable/distributed.html""")
    # parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.") # Orignal code
    parser.add_argument("--local-rank", default=os.environ['LOCAL_RANK'], type=int, help="Please ignore and do not set this argument.") # Adapted code

    return parser


''' Original Data Augmentation '''
# class DataAugmentationDINO(object):

#     def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):

#         flip_and_color_jitter = transforms.Compose([
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
#             transforms.RandomGrayscale(p=0.2),
#         ])
#         normalize = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ])

#         # First Global Crop
#         self.global_transfo1 = transforms.Compose([
#             transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
#             flip_and_color_jitter,
#             utils.GaussianBlur(1.0),
#             normalize,
#         ])

#         # Second Global Crop
#         self.global_transfo2 = transforms.Compose([
#             transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
#             flip_and_color_jitter,
#             utils.GaussianBlur(0.1),
#             utils.Solarization(0.2),
#             normalize,
#         ])

#         # Transformation for the local small crops
#         self.local_crops_number = local_crops_number
#         self.local_transfo = transforms.Compose([
#             transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
#             flip_and_color_jitter,
#             utils.GaussianBlur(p=0.5),
#             normalize,
#         ])


###########################
''' Data Augmentation 1 '''
###########################
class DataAugmentationDINO(object):

    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, mean=None, std=None):

        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5), # Additionally added
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.0, hue=0.0)], p=0.8)
        ])

        if (mean is not None) and (std is not None):
            normalize = transforms.Compose([
                transforms.Normalize(mean=mean, std=std)
                # For image tensors with values in [0.0, 1.0] this transformation will standardize it,
                # so that the mean of the data should be ~0 and the std ~1
                # (See Standard Score: https://en.wikipedia.org/wiki/Standard_score)
                # Note: Before transforms.ToTensor() was applied and ensured that the tensor is in range [0.0, 1.0]
            ])
        else:
            print("\nWarning: Please specify values for [mean] and [std] if normalization should be used!")
            normalize = transforms.Compose([])

        # First Global Crop
        self.global_transfo1 = transforms.Compose([
            # transforms.RandomResizedCrop(size=224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            transforms.RandomResizedCrop(size=224, scale=global_crops_scale, ratio=(1.0, 1.0), interpolation=InterpolationMode.BICUBIC), # [ratio] was additionally added
            flip_and_color_jitter,
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5,5), sigma=(0.1,2.0))], p=1.0), # What is a good choice for [kernel_size]?
            normalize,
        ])

        # Second Global Crop
        self.global_transfo2 = transforms.Compose([
            # transforms.RandomResizedCrop(size=224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            transforms.RandomResizedCrop(size=224, scale=global_crops_scale, ratio=(1.0, 1.0), interpolation=InterpolationMode.BICUBIC), # [ratio] was additionally added
            flip_and_color_jitter,
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5,5), sigma=(0.1,2.0))], p=0.1), # What is a good choice for [kernel_size]?
            transforms.RandomSolarize(p=0.2, threshold=0.5), # Default threshold for PIL images with range [0, 256] was 128
            normalize,
        ])

        # Transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            # transforms.RandomResizedCrop(size=96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            transforms.RandomResizedCrop(size=96, scale=local_crops_scale, ratio=(1.0, 1.0), interpolation=InterpolationMode.BICUBIC), # [ratio] was additionally added
            flip_and_color_jitter,
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5,5), sigma=(0.1,2.0))], p=0.5), # What is a good choice for [kernel_size]?
            normalize,
        ])


    def __call__(self, image):

        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))

        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))

        return crops # List containing multiple torch tensors
    

###########################
''' Data Augmentation 2 '''
###########################
class DataAugmentationV2(object):

    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, mean=None, std=None):

        print("\nWarning: Using custom data augmentation class 'DataAugmentationV2()' with random neglog transform and rotation!'")

        class RandomAdjustSharpness:
            def __init__(self, sharpness_factor=0.2, p=1.0):
                self.sharpness_factor = sharpness_factor
                self.p = p

            def __call__(self, img):
                sharpness_factor = random.uniform(1-self.sharpness_factor, 1+self.sharpness_factor) # e.g., [1-0.2, 1+0.2] = [0.8, 1.2]
                return transforms.functional.adjust_sharpness(img, sharpness_factor) if random.random() < self.p else img
                # sharpness_factor – How much to adjust the sharpness. Can be any non-negative number.
                # 0 gives a blurred image, 1 gives the original image while 2 increases the sharpness by a factor of 2.

        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5), # Additionally added
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.0, hue=0.0)
                ], p=0.8)
        ])

        if (mean is not None) and (std is not None):
            normalize = transforms.Compose([
                transforms.Normalize(mean=mean, std=std)
                # For image tensors with values in [0.0, 1.0] this transformation will standardize it,
                # so that the mean of the data should be ~0 and the std ~1
                # (See Standard Score: https://en.wikipedia.org/wiki/Standard_score)
                # Note: Before transforms.ToTensor() was applied and ensured that the tensor is in range [0.0, 1.0]
            ])
        else:
            print("\nWarning: Please specify values for [mean] and [std] if normalization should be used!")
            normalize = transforms.Compose([])

        # First Global Crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=global_crops_scale, ratio=(1.0, 1.0), interpolation=InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            RandomAdjustSharpness(sharpness_factor=0.2, p=1.0),
            normalize,
        ])

        # Second Global Crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomApply([
                transforms.RandomRotation(degrees=(-180, 180), interpolation=InterpolationMode.BILINEAR, expand=False, center=None, fill=0)
                ], p=0.8),
            transforms.RandomResizedCrop(size=224, scale=global_crops_scale, ratio=(1.0, 1.0), interpolation=InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            RandomAdjustSharpness(sharpness_factor=0.2, p=1.0),
            transforms.RandomSolarize(p=0.2, threshold=0.5), # Default threshold for PIL images with range [0, 256] was 128
            normalize,
        ])

        # Local Crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomApply([
                transforms.RandomRotation(degrees=(-180, 180), interpolation=InterpolationMode.BILINEAR, expand=False, center=None, fill=0)
                ], p=0.8),
            transforms.RandomResizedCrop(size=96, scale=local_crops_scale, ratio=(1.0, 1.0), interpolation=InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            RandomAdjustSharpness(sharpness_factor=0.2, p=1.0),
            normalize,
        ])


    def __call__(self, image):

        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))

        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))

        return crops # List containing multiple torch tensors
    

class DINOLoss(nn.Module):

    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs,
                 nepochs, student_temp=0.1, center_momentum=0.9):
        super().__init__()

        self.ncrops = ncrops
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

        # We apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning.
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp))

    def forward(self, student_output, teacher_output, epoch):

        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(chunks=self.ncrops) # Attempts to split a tensor into the specified number of chunks.

        # Teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(chunks=2) # Attempts to split a tensor into the specified number of chunks.
        
        """ Original Code """

        # total_loss = 0
        # n_loss_terms = 0
        # for iq, q in enumerate(teacher_out):
        #     for v in range(len(student_out)):
        #         if v == iq:
        #             # We skip cases where student and teacher operate on the same view
        #             continue
        #         loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
        #         total_loss += loss.mean()
        #         n_loss_terms += 1
        # total_loss /= n_loss_terms
        # self.update_center(teacher_output)

        # return total_loss

        """ Customized Code """
        total_loss = 0
        n_loss_terms = 0
        student_entropy, teacher_entropy, kl_div = 0, 0, 0
        for ip, p in enumerate(teacher_out):
            for v in range(len(student_out)):

                if v == ip:
                    # We skip cases where student and teacher operate on the same view
                    # (i.e., the first two global views)
                    continue

                loss = torch.sum(-p * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
                
                # Set small constant to prevent log(0) and division by 0
                epsilon = 1e-10

                # Student Entropy
                student_prob = F.softmax(student_out[v].detach(), dim=-1) # [B, output_dim]
                student_prob = torch.clamp(student_prob, min=epsilon)
                student_entropy += torch.mean(-1.0 * torch.sum(student_prob * torch.log(student_prob), dim=-1))

                # Teacher Entropy
                teacher_prob = p.detach() # [B, output_dim]
                teacher_prob = torch.clamp(teacher_prob, min=epsilon)
                teacher_entropy += torch.mean(-1.0 * torch.sum(teacher_prob * torch.log(teacher_prob), dim=-1))

                # KL-Divergence
                # This measures how the student network's distribution diverges from the teacher network's distribution,
                # which is useful for training the student network to mimic the teacher network more closely.
                kl_div += torch.mean(torch.sum(teacher_prob * torch.log(teacher_prob / student_prob), dim=-1))

        total_loss /= n_loss_terms
        self.update_center(teacher_output)

        student_entropy /= n_loss_terms
        teacher_entropy /= n_loss_terms
        kl_div /= n_loss_terms

        loss_metrics = {
            'student_entropy'   : student_entropy,
            'teacher_entropy'   : teacher_entropy,
            'kl_divergence'     : kl_div,
            'teacher_temp'      : temp,
            'student_temp'      : self.student_temp
        }
    
        return total_loss, loss_metrics

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # EMA update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


def train_dino(args):

    # If-condition is needed since setup for torch.distributed has not yet been called!
    if int(os.environ["RANK"]) == 0: # Check the global rank
        print("\nEntered training method ...")
        env_variables = ["LOCAL_RANK", "RANK", "LOCAL_WORLD_SIZE", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
        for var in env_variables:
            print(f"{var} = {os.environ[var]}")
        print() # Just print an empty line ...
    
    utils.init_distributed_mode(args) # Initialize parameters for torch.distributed
    utils.fix_random_seeds(args.seed) # Fix random seeds for torch, cuda, and numpy

    ###############################################################################
    # dist.barrier() # Additionally added
    # ''' The following function is called at the start of each training to ensure
    # that the main process has a CUDA context w.r.t. all other GPUs initialized
    # before calling 'get_gpu_info'. This is important for memory analysis. '''
    # if utils.is_main_process():
    #     utils.initialize_cuda_context() # Additionally added
    ###############################################################################

    print(f"\nGIT Information\n{utils.get_sha()}\n")
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    
    # Setup for W&B logging
    if args.use_wandb and utils.is_main_process():
        try:
            print() # Just print an empty line ...
            api_key = os.environ["WANDB_API_KEY"]
            wandb.login(key=api_key)
            print(f"\nSuccessfully logged in to W&B with API key '{api_key}'\n")
        except Exception as e:
            print(f"\nCould not login to W&B service: {e}\n")

        if args.arch == "resnet18": group = "ResNet18 Models"
        elif args.arch == "resnet50": group = "ResNet50 Models"
        elif args.arch == "vit_tiny": group = "ViT-T Models"
        elif args.arch == "vit_small": group = "ViT-S Models"
        else: group = "Misc Models"

        if args.azure: # Azure ML Studio
            wandb_dir = args.output_dir
        else: # local Cluster
            wandb_dir = "/home/local/path"

        wandb.init(project = "project-name",
                   entity = "user-name",
                   config = args, # save settings and hyperparameters
                   save_code = True, # save main script
                   group = group, # organize individual runs into a larger experiment
                   name = args.job_ID, # general job descriptor
                   id = args.job_ID, # unique job descriptor
                   resume = "allow", # needed in case of preempted Azure job
                   dir = wandb_dir)
    
    # Setup for Azure logging
    if args.azure:
        run = Run.get_context()

    """
    A bool that, if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
    It enables benchmark mode in cudnn, which is good whenever your input sizes for your network do not vary.
    This way, cudnn will look for the optimal set of algorithms for that particular configuration (which takes some time).
    It usually leads to faster runtime, but if your input sizes change at each iteration, then cudnn will benchmark
    every time a new size appears, possibly leading to worse runtime performances ...
    """
    cudnn.benchmark = True

    if utils.is_main_process(): # Additionally added
        
        print() # Just an empty line ...
        os.system("nvidia-smi") # Call 'nvidia-smi' on the main process

        mem_buffer = utils.Memory_Buffer()
        gpu_info = mem_buffer("Initialization of Distributed Mode")
        utils.get_shm_size()

    #####################################################################
    # ======================== Preparing Data ===========================
    #####################################################################
    """ Note: Always specify the correct values for mean and std """

    if args.augmentation == "v1":
        transforms = DataAugmentationDINO(
            global_crops_scale = args.global_crops_scale,
            local_crops_scale = args.local_crops_scale,
            local_crops_number = args.local_crops_number,
            mean = args.data_mean, # Update this parameter according to the used dataset!
            std = args.data_std) # Update this parameter according to the used dataset!
        
    elif args.augmentation == "v2":
        transforms = DataAugmentationV2(
            global_crops_scale = args.global_crops_scale,
            local_crops_scale = args.local_crops_scale,
            local_crops_number = args.local_crops_number,
            mean = args.data_mean, # Update this parameter according to the used dataset!
            std = args.data_std) # Update this parameter according to the used dataset!
    
    """ %%%%%%%%%%%%%%%%%%%%%%% Original Code %%%%%%%%%%%%%%%%%%%%%%% """
    # dataset = datasets.ImageFolder(args.data_path, transform=transform)

    """ %%%%%%%%%%%%%%%%%%%%%% Customized Code %%%%%%%%%%%%%%%%%%%%%% """
    if args.azure: # Azure ML Studio
        if args.dataset.startswith("DINO-Dataset-v"):
            if args.augmentation == "v1":
                print(f"\nUsing dataset class 'AzureDataset()' with subtract_lowpass = {args.subtract_lowpass}")
                dataset = AzureDataset(dataset_path = args.data_path,
                                       subtract_lowpass = args.subtract_lowpass,
                                       transforms = transforms)
            elif args.augmentation == "v2":
                print(f"\nUsing dataset class 'AzureDatasetV2()' with subtract_lowpass = {args.subtract_lowpass}")
                dataset = AzureDatasetV2(dataset_path = args.data_path,
                                         subtract_lowpass = args.subtract_lowpass,
                                         transforms = transforms)
        else:
            if args.augmentation == "v1":
                print(f"\nUsing dataset class 'DummyDataset()' on Azure Cluster with subtract_lowpass = {args.subtract_lowpass}")
                dataset = DummyDataset(path = args.data_path,
                                       subtract_lowpass = args.subtract_lowpass,
                                       transforms = transforms)
            elif args.augmentation == "v2":
                print(f"\nUsing dataset class 'DummyDatasetV2()' on Azure Cluster with subtract_lowpass = {args.subtract_lowpass}")
                dataset = DummyDatasetV2(path = args.data_path,
                                         subtract_lowpass = args.subtract_lowpass,
                                         transforms = transforms)
    else: # Local Cluster
        if args.augmentation == "v1":
            print(f"\nUsing dataset class 'DummyDataset()' on local Cluster with subtract_lowpass = {args.subtract_lowpass}")
            dataset = DummyDataset(path = args.data_path,
                                   subtract_lowpass = args.subtract_lowpass,
                                   transforms = transforms)
        elif args.augmentation == "v2":
            print(f"\nUsing dataset class 'DummyDatasetV2()' on local Cluster with subtract_lowpass = {args.subtract_lowpass}")
            dataset = DummyDatasetV2(path = args.data_path,
                                     subtract_lowpass = args.subtract_lowpass,
                                     transforms = transforms)
    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """

    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        # Parameter [num_workers]: How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
        pin_memory=True,
        # Parameter [pin_memory]: If True, the dataloader will copy Tensors into device/CUDA pinned memory before returning them.
        # With pinned memory tensors you can use x.cuda(non_blocking=True) to perform the copy asynchronously with respect to host.
        # This can lead to performance gains in certain scenarios.
        drop_last=True,
        # Parameter [drop_last]: Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size.
        prefetch_factor=2 # Additionally added
        # Parameter [prefetch_factor]: Number of batches loaded in advance by each worker.
        # 3 means there will be a total of 3 * num_workers batches prefetched. 
        )

    print(f"\nData loaded: There are {len(dataset)} single projection images available.")

    if utils.is_main_process(): # Additionally added
        gpu_info = mem_buffer("Dataloader Setup")
        utils.get_shm_size()
    
    #####################################################################
    # ============= Building Student and Teacher Networks ===============
    #####################################################################

    # We changed the name DeiT-S for ViT-S to avoid confusions ...
    """
    DeiT (Data Efficient Image Transformer) has the same architecture as ViT except the input token part
    that having an additional distillation token. But since ViT cannot perform well when trained on
    insufficient amounts of data. Distillation is a way to train.
    """
    args.arch = args.arch.replace("deit", "vit")

    # If the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate) # stochastic depth
        teacher = vits.__dict__[args.arch](
            patch_size=args.patch_size)
        embed_dim = student.embed_dim
    # Otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    """
    The following code was enabled in the original implementation, but disabled for now due to bug fixes (see argparser).
    Insert this condition again after the first 'if' statement ...
    """
    # If the network is a XCiT (Cross-Covariance Image Transformer)
    # elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
    #     student = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False, drop_path_rate=args.drop_path_rate)
    #     teacher = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False)
    #     embed_dim = student.embed_dim

    ''' %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% '''
    '''                             Original code                              '''
    ''' %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% '''
    
    # # Multi-crop wrapper handles forward with inputs of different resolutions
    # student = utils.MultiCropWrapper(
    #     student,
    #     DINOHead(embed_dim, args.out_dim, use_bn=args.use_bn_in_head, norm_last_layer=args.norm_last_layer))
    # teacher = utils.MultiCropWrapper(
    #     teacher,
    #     DINOHead(embed_dim, args.out_dim, args.use_bn_in_head))
    
    ''' %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% '''
    ''' The following code was additionally added to support model pretraining '''
    ''' %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% '''

    if args.pretrained_weights != 'No Pretraining': # Only for ImageNet pretraining ...

        if (args.arch == 'resnet50'):
            student = utils.MultiCropWrapper(
                student,
                DINOHead(embed_dim, args.out_dim, use_bn=args.use_bn_in_head, norm_last_layer=args.norm_last_layer,
                        hidden_dim=4096, bottleneck_dim=256, nlayers=2, pretrained=True)) # Added for compatibility
            teacher = utils.MultiCropWrapper(
                teacher,
                DINOHead(embed_dim, args.out_dim, args.use_bn_in_head,
                        hidden_dim=4096, bottleneck_dim=256, nlayers=2, pretrained=True)) # Added for compatibility
    
        elif args.arch == 'vit_small':
            student = utils.MultiCropWrapper(
                student,
                DINOHead(embed_dim, args.out_dim, use_bn=args.use_bn_in_head, norm_last_layer=args.norm_last_layer))
            teacher = utils.MultiCropWrapper(
                teacher,
                DINOHead(embed_dim, args.out_dim, args.use_bn_in_head))
            
        else:
            raise ValueError(f"Pretraining for {args.arch} architecture is not supported yet!")
            
        print(f"\nTry to use pretrained weights for {args.arch} architecture:")

        utils.load_pretrained_weights(model=student,
                                      checkpoint_key='student',
                                      pretrained_weights=args.pretrained_weights,
                                      model_name=args.arch,
                                      patch_size=args.patch_size)
        
        utils.load_pretrained_weights(model=teacher,
                                      checkpoint_key='teacher',
                                      pretrained_weights=args.pretrained_weights,
                                      model_name=args.arch,
                                      patch_size=args.patch_size)
        
        # Note that in addtion also utils.restart_from_checkpoint()
        # is called before the actual start of training!

    else:
        
        student = utils.MultiCropWrapper(
            student,
            DINOHead(embed_dim, args.out_dim, use_bn=args.use_bn_in_head, norm_last_layer=args.norm_last_layer))
        teacher = utils.MultiCropWrapper(
            teacher,
            DINOHead(embed_dim, args.out_dim, args.use_bn_in_head))
        
        print(f"\nNo use of pretrained weights for {args.arch} architecture!")
        
    ''' %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% '''

    # Move networks to GPU
    student, teacher = student.cuda(), teacher.cuda()

    ''' %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% '''
    ''' The following code was additionally added '''
    if utils.is_main_process():
        print(f"\nTotal number of parameters for a single [Student-Backbone]: {utils.num_parameters(student.backbone)} Million")
        print(f"Total number of parameters for a single [Student-Head]: {utils.num_parameters(student.head)} Million")
        print(f"Total number of parameters for a single [Teacher-Backbone]: {utils.num_parameters(teacher.backbone)} Million")
        print(f"Total number of parameters for a single [Teacher-Head]: {utils.num_parameters(teacher.head)} Million")
        gpu_info = mem_buffer("Model Setup")
        utils.get_shm_size()
    ''' %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% '''

    # Synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # We need DDP wrapper to have synchro batch norms working ...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
        
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])

    # Teacher and Student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())

    # There is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"\nStudent and Teacher are built: They are both {args.arch} networks.")

    #####################################################################
    # ========================= Preparing Loss ==========================
    #####################################################################
    dino_loss = DINOLoss(
        out_dim = args.out_dim,
        ncrops = args.local_crops_number + 2, # total number of crops = 2 global crops + local_crops_number
        warmup_teacher_temp = args.warmup_teacher_temp,
        teacher_temp = args.teacher_temp,
        warmup_teacher_temp_epochs = args.warmup_teacher_temp_epochs,
        nepochs = args.epochs
    ).cuda()

    #####################################################################
    # ======================= Preparing Optimizer =======================
    #####################################################################
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups) # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9) # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups) # to use with convnet and large batches

    #####################################################################
    # ==================== Mixed Precision Training =====================
    #####################################################################
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.amp.GradScaler("cuda") # Adapted code
        # fp16_scaler = torch.cuda.amp.GradScaler() # Original code

    #####################################################################
    # ========================= Init Schedulers =========================
    #####################################################################
    lr_schedule = utils.cosine_scheduler(
        base_value = args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
        final_value = args.min_lr,
        epochs = args.epochs,
        niter_per_ep = len(data_loader),
        warmup_epochs = args.warmup_epochs)

    wd_schedule = utils.cosine_scheduler(
        base_value = args.weight_decay,
        final_value = args.weight_decay_end,
        epochs = args.epochs,
        niter_per_ep = len(data_loader))

    # Momentum parameter is increased to 1.0 during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(
        base_value = args.momentum_teacher,
        final_value = 1,
        epochs = args.epochs,
        niter_per_ep = len(data_loader))
    
    print(f"\nLoss, optimizer and schedulers ready.")

    #####################################################################
    # =================== Optionally resume training ====================
    #####################################################################
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        ckp_path = os.path.join(args.output_dir, "backup_checkpoint.pth"),
        run_variables = to_restore,
        student = student,
        teacher = teacher,
        optimizer = optimizer,
        dino_loss = dino_loss,
        fp16_scaler = fp16_scaler)
    # Note: 'to_restore' gets updated from the last checkpoint including {"epoch" = epoch + 1}
    start_epoch = to_restore["epoch"]

    if utils.is_main_process(): # Additionally added
        gpu_info = mem_buffer("Training Start")
        utils.get_shm_size()

    #####################################################################
    # ====================== Start DINO Training ========================
    #####################################################################
    start_time = time.time()
    print("\nStarting DINO training!")
    for epoch in range(start_epoch, args.epochs):

        data_loader.sampler.set_epoch(epoch)

        # ============ Training one epoch of DINO ============
        train_stats = train_one_epoch(
            student, teacher, teacher_without_ddp, dino_loss, data_loader, optimizer,
            lr_schedule, wd_schedule, momentum_schedule, epoch, fp16_scaler, args)

        # ============ Writing logs ==========================
        # Note: Do not change the key names of this dictionary!
        save_dict = {
            'student'   : student.state_dict(),
            'teacher'   : teacher.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'dino_loss' : dino_loss.state_dict(),
            'epoch'     : epoch+1,  # This ensures that the training will be resumed at the next epoch and
                                    # that the epoch value is consistent w.r.t. other logfiles and checkpoints.
            'args'      : args
            }

        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()

        utils.save_on_master(save_dict, os.path.join(args.output_dir, "backup_checkpoint.pth")) # Checkpoint to restart training

        if args.saveckp_freq and (((epoch+1) % args.saveckp_freq) == 0 or (epoch+1) == args.epochs): # Adapted code
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f"checkpoint_{args.arch}_{args.job_ID}_{epoch+1:04}.pth"))
        
        log_stats = {'epoch': epoch+1, **{f'train_{k}': v for k, v in train_stats.items()}}
        if utils.is_main_process():

            with (Path(args.output_dir) / f"{args.job_ID}_log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            wandb.log(log_stats, commit=False) # Additionally added W&B logging

            gpu_info = mem_buffer(f"Epoch {epoch+1}") # Additionally added memory logging
            wandb.log(gpu_info, commit=True) # Additionally added W&B logging
            utils.get_shm_size()

            if args.azure: # Additionally added Azure logging
                for k, v in log_stats.items():
                    run.log(name=k, value=v)

    training_time = time.time() - start_time
    training_time_str = str(datetime.timedelta(seconds=int(training_time)))
    print('\nTraining Time = {}\n'.format(training_time_str))

    # Finish W&B run
    if utils.is_main_process():
        wandb.finish(quiet=True)


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader, optimizer,
                    lr_schedule, wd_schedule, momentum_schedule, epoch, fp16_scaler, args):
    
    print("") # Add empty line between each single epoch
    metric_logger = utils.MetricLogger(delimiter=" - ")
    header = 'Epoch: [{}/{}]'.format(epoch+1, args.epochs)
    for it, (images, _) in enumerate(metric_logger.log_every(iterable=data_loader, print_freq=10, header=header)):

        # Update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it # Global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0: # Only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # Move images to GPU
        images = [im.cuda(non_blocking=True) for im in images]

        # Teacher and student forward passes + Compute DINO loss
        with torch.amp.autocast(device_type="cuda", enabled = fp16_scaler is not None): # Adapted code
        # with torch.cuda.amp.autocast(fp16_scaler is not None): # Original code

            teacher_output = teacher(images[:2]) # Only the 2 global views pass through the teacher

            student_output = student(images) # Is this correct that student gets all views (i.e., local AND global)???

            loss, loss_metrics = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # Student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer) # Unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it] # Momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # Logging
        torch.cuda.synchronize() # Ensure completion of CUDA operations (wait for them to finish)
        """
        Use dist.barrier() when you need to synchronize multiple processes in a distributed training.

        Use torch.cuda.synchronize() when you need to ensure that GPU computations are completed before the CPU proceeds,
        typically for timing, logging or correctness purposes within a single process.

        """
        metric_logger.update(loss = loss.item())
        metric_logger.update(student_entropy = loss_metrics['student_entropy'].item()) # Additionally added
        metric_logger.update(teacher_entropy = loss_metrics['teacher_entropy'].item()) # Additionally added
        metric_logger.update(kl_divergence = loss_metrics['kl_divergence'].item()) # Additionally added
        metric_logger.update(lr = optimizer.param_groups[0]["lr"])
        metric_logger.update(wd = optimizer.param_groups[0]["weight_decay"])
        metric_logger.update(student_temp = loss_metrics['student_temp']) # Additionally added
        metric_logger.update(teacher_temp = loss_metrics['teacher_temp']) # Additionally added
        metric_logger.update(momentum = m) # Additionally added

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':

    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # os.makedirs(args.output_dir, exist_ok=True) # Just an alternative implementation ...

    output_file = Path(args.output_dir) / f"{args.job_ID}_settings.json"
    with open(output_file, 'w') as f:
        json.dump(vars(args), f, indent=4)

    # If-condition is needed since setup for torch.distributed has not yet been called!
    if int(os.environ["RANK"]) == 0: # Check the global rank
        # [offline] will save your run metadata locally and not sync to the server
        # [disabled] will turn off collecting metadata completely
        if args.use_wandb == False:
            os.environ["WANDB_MODE"] = "disabled"
            print("\nInfo: W&B logging is disabled!")
        else:
            print("\nInfo: W&B is enabled!")

    train_dino(args)
