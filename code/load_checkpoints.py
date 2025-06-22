from pathlib import Path

import utils

import torch
import torch.nn as nn

import torchvision.models as models
import vision_transformer_dax as vits


##################################################################################
model_type = "ResNet50"
checkpoint_file = "path/to/local/checkpoint/dax-checkpoint-resnet50-version-a.pth"
##################################################################################

if model_type in ["ResNet18", "ResNet50"]:

    checkpoint = torch.load(Path(checkpoint_file), map_location="cpu", weights_only=False)
    teacher_checkpoint = checkpoint["teacher"]
    # Discard all weights and parameters belonging to DINOHead() ...
    teacher_dict = {k.replace("module.backbone.", ""): v for k, v in teacher_checkpoint.items() if k.startswith("module.backbone.")}

    assert model_type.lower() in models.__dict__.keys()
    resnet = models.__dict__[model_type.lower()](weights=None)
    embed_dim = resnet.fc.weight.shape[1]
    resnet.fc = nn.Identity() # Needed to successfully load checkpoint
    msg = resnet.load_state_dict(teacher_dict, strict=True)
    print(f"\nPretrained weights found at '{checkpoint_file}'\nand loaded with msg: {msg}")
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1]) # Discard the last FC layer (only needed for class predictions)

    model = resnet
    model = model.cuda()

elif model_type in ["ViT-T-16", "ViT-T-8"]:

    checkpoint = torch.load(Path(checkpoint_file), map_location="cpu", weights_only=False)
    teacher_checkpoint = checkpoint['teacher']
    # Discard all weights and parameters belonging to DINOHead() ...
    teacher_dict = {k.replace('backbone.', ''): v for k, v in teacher_checkpoint.items() if k.startswith('backbone.')}

    patch_size = int(model_type.split("-")[-1])
    vit_tiny = vits.vit_tiny(patch_size=patch_size)
    msg = vit_tiny.load_state_dict(teacher_dict, strict=True)
    print(f"\nPretrained weights found at '{checkpoint_file}'\nand loaded with msg: {msg}")
    
    model = vit_tiny
    model = model.cuda()

elif model_type in ["ViT-S-16", "ViT-S-8"]:

    checkpoint = torch.load(Path(checkpoint_file), map_location="cpu", weights_only=False)
    teacher_checkpoint = checkpoint['teacher']
    # Discard all weights and parameters belonging to DINOHead() ...
    teacher_dict = {k.replace('backbone.', ''): v for k, v in teacher_checkpoint.items() if k.startswith('backbone.')}

    patch_size = int(model_type.split("-")[-1])
    vit_small = vits.vit_small(patch_size=patch_size)
    msg = vit_small.load_state_dict(teacher_dict, strict=True)
    print(f"\nPretrained weights found at '{checkpoint_file}'\nand loaded with msg: {msg}")
    
    model = vit_small
    model = model.cuda()

print(f"\nPre-trained model has {utils.num_parameters(model)} million parameters!")
