import torch.nn as nn
import os
import torch
import torch.nn.functional as F
import numpy as np

from torchvision.models.video import MC3_18_Weights, mc3_18
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from torchvision.ops import SqueezeExcitation

from models_utilities import remove_module_prefix, rename_module_prefix, count_parameters
from resnet_mag.magnification.mag_model import load_network




def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


class InfuseNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.spatial_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).cuda()
        self.spatial_model_mag = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).cuda()
        linear_layer_temp = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        linear_layer = nn.Linear(in_features=512, out_features=num_classes, bias=True)

        # weights freezing
        for param in self.spatial_model.parameters():
            param.requires_grad = True      
        for param in self.spatial_model.layer4.parameters():
            param.requires_grad = True    

        for param in self.spatial_model_mag.parameters():
            param.requires_grad = True      
        for param in self.spatial_model_mag.layer4.parameters():
            param.requires_grad = True    

        
        self.conv1 = self.spatial_model.conv1
        self.bn1 = self.spatial_model.bn1
        self.relu = self.spatial_model.relu
        self.maxpool = self.spatial_model.maxpool
        self.layer1 = self.spatial_model.layer1
        self.layer2 = self.spatial_model.layer2
        self.layer3 = self.spatial_model.layer3
        self.layer4 = self.spatial_model.layer4
        self.avgpool = self.spatial_model.avgpool
        self.flatten = nn.Flatten()
        self.fc = linear_layer     

        self.spatial_model.fc = linear_layer
        self.spatial_model_mag.fc = linear_layer

        # Magnification Model
        checkpoint_path = '/home/hq/Documents/Weights/Magnification/generator_212000.pth'
        mag_model = load_network(checkpoint_path)
        mag_model = mag_model.cuda()

        self.mag_encoder = mag_model.encoder
        self.mag_manipulator = mag_model.manipulator
        self.mag_decoder = mag_model.decoder

    def forward(self, x, x_rgb_onset, x_rgb_apex, x_window_apex):

        # test-time
        amp_one = 10
        amp_two = 10   

        # magnifying annotated onset and annotated apex
        x_mag = self.mag_encoder(x_rgb_onset) # (32, 32, 112, 112)
        x_offset_mag = self.mag_encoder(x_rgb_apex) # (32, 32, 112, 112)
        magnified_diff, _ = self.mag_manipulator(x_mag, x_offset_mag, amp=amp_one) # (32, 32, 112, 112)

        # magnifying annotated onset and approx-window apex
        x_mag = self.mag_encoder(x_rgb_onset) # (32, 32, 112, 112)
        x_offset_mag = self.mag_encoder(x_window_apex) # (32, 32, 112, 112)
        magnified_diff_approx, _ = self.mag_manipulator(x_mag, x_offset_mag, amp=amp_two) # (32, 32, 112, 112)

        # concat both of them
        magnified_diff_concat = torch.cat([magnified_diff, magnified_diff_approx], dim=1)

        # Experiment E1
        # res18 for optical flow
        x_res = self.conv1(x) # (32, 64, 112, 112)
        x_res = self.bn1(x_res) 
        x_res = self.relu(x_res)
        x_res = self.maxpool(x_res) # (32, 64, 16, 16) # can i magnify here?
        x_res_layer1 = self.layer1(x_res) # (32, 64, 16, 16) # can i magnify here?
        norm_x_res_layer1 = (x_res_layer1 - x_res_layer1.amin()) / (x_res_layer1.amax() - x_res_layer1.amin())
        x_res_layer2 = self.layer2(x_res_layer1) # (32, 128, 8, 8)
        norm_x_res_layer2 = (x_res_layer2 - x_res_layer2.amin()) / (x_res_layer2.amax() - x_res_layer2.amin())
        x_res_layer3 = self.layer3(x_res_layer2) # (32, 256, 4, 4)
        norm_x_res_layer3 = (x_res_layer3 - x_res_layer3.amin()) / (x_res_layer3.amax() - x_res_layer3.amin())
        x_res_layer4 = self.layer4(x_res_layer3)  # (32, 512, 2, 2)
        norm_x_res_layer4 = (x_res_layer4 - x_res_layer4.amin()) / (x_res_layer4.amax() - x_res_layer4.amin())
        x_res_avgpool = self.avgpool(x_res_layer4) # (32, 512, 1, 1)
        norm_x_res_avgpool = (x_res_avgpool - x_res_avgpool.amin()) / (x_res_avgpool.amax() - x_res_avgpool.amin())
        x_res = self.flatten(x_res_avgpool) # add a flatten 

        x_mag = self.spatial_model_mag.relu(magnified_diff_concat)
        x_mag_maxpool = self.spatial_model_mag.maxpool(x_mag)
        x_mag_layer1 = self.spatial_model_mag.layer1(x_mag_maxpool) * norm_x_res_layer1 # (32, 64, 16, 16) # can i magnify here?
        x_mag_layer2 = self.spatial_model_mag.layer2(x_mag_layer1) * norm_x_res_layer2 # (32, 128, 8, 8)
        x_mag_layer3 = self.spatial_model_mag.layer3(x_mag_layer2) * norm_x_res_layer3 # (32, 256, 4, 4)
        x_mag_layer4 = self.spatial_model_mag.layer4(x_mag_layer3) * norm_x_res_layer4 # (32, 512, 2, 2)
        x_mag_avgpool = self.spatial_model_mag.avgpool(x_mag_layer4) * norm_x_res_avgpool # (32, 512, 1, 1)
        x_mag = self.flatten(x_mag_avgpool) # add a flatten   

        x_res = self.fc(x_res)
        x_mag = self.spatial_model_mag.fc(x_mag)

        return x_res, x_mag

class InfuseNet_for_GradCAM(nn.Module):
    def __init__(self, num_classes: int, weights_path: str):
        super().__init__()

        self.spatial_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).cuda()
        self.spatial_model_mag = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).cuda()
        linear_layer_temp = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        linear_layer = nn.Linear(in_features=512, out_features=num_classes, bias=True)    

        # weights freezing
        for param in self.spatial_model.parameters():
            param.requires_grad = True      
        for param in self.spatial_model.layer4.parameters():
            param.requires_grad = True    

        for param in self.spatial_model_mag.parameters():
            param.requires_grad = True      
        for param in self.spatial_model_mag.layer4.parameters():
            param.requires_grad = True    

        
        self.conv1 = self.spatial_model.conv1
        self.bn1 = self.spatial_model.bn1
        self.relu = self.spatial_model.relu
        self.maxpool = self.spatial_model.maxpool
        self.layer1 = self.spatial_model.layer1
        self.layer2 = self.spatial_model.layer2
        self.layer3 = self.spatial_model.layer3
        self.layer4 = self.spatial_model.layer4
        self.avgpool = self.spatial_model.avgpool
        self.flatten = nn.Flatten()
        self.fc = linear_layer     

        self.spatial_model.fc = linear_layer
        self.spatial_model_mag.fc = linear_layer


    def forward(self, x):
        # Experiment E1
        # res18 for optical flow
        x_res = self.conv1(x) # (32, 64, 112, 112)
        x_res = self.bn1(x_res) 
        x_res = self.relu(x_res)
        x_res = self.maxpool(x_res) # (32, 64, 16, 16) # can i magnify here?
        x_res_layer1 = self.layer1(x_res) # (32, 64, 16, 16) # can i magnify here?
        norm_x_res_layer1 = (x_res_layer1 - x_res_layer1.amin()) / (x_res_layer1.amax() - x_res_layer1.amin())
        x_res_layer2 = self.layer2(x_res_layer1) # (32, 128, 8, 8)
        norm_x_res_layer2 = (x_res_layer2 - x_res_layer2.amin()) / (x_res_layer2.amax() - x_res_layer2.amin())
        x_res_layer3 = self.layer3(x_res_layer2) # (32, 256, 4, 4)
        norm_x_res_layer3 = (x_res_layer3 - x_res_layer3.amin()) / (x_res_layer3.amax() - x_res_layer3.amin())
        x_res_layer4 = self.layer4(x_res_layer3)  # (32, 512, 2, 2)
        norm_x_res_layer4 = (x_res_layer4 - x_res_layer4.amin()) / (x_res_layer4.amax() - x_res_layer4.amin())
        x_res_avgpool = self.avgpool(x_res_layer4) # (32, 512, 1, 1)
        norm_x_res_avgpool = (x_res_avgpool - x_res_avgpool.amin()) / (x_res_avgpool.amax() - x_res_avgpool.amin())
        x_res = self.flatten(x_res_avgpool) # add a flatten 
        x_res = self.fc(x_res)
        return x_res


