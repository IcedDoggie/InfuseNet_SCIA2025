import argparse
import os
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image

from resnet_mag.magnification.magnify.configuration import denorm
from resnet_mag.magnification.magnify.generator import MagNet

# from magnify.configuration import denorm
# from magnify.generator import MagNet


def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove 'module.' prefix
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict




def load_network(checkpoint_path):

    model_path = checkpoint_path
    # create model
    model = MagNet()
    # model = torch.nn.DataParallel(model).cuda()

    # load checkpoint
    # 从预训练好的模型中加载参数
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        checkpoint = remove_module_prefix(checkpoint)

        model.load_state_dict(checkpoint, strict=True)
        print("=> loaded checkpoint '{}' "
              .format(model_path))
    else:
        print("=> no checkpoint found at '{}'".format(model_path))


    # cudnn enable
    cudnn.benchmark = True

    return model

