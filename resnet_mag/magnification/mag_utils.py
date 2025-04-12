import argparse
import os
import matplotlib.pyplot as plt
from natsort import natsorted


import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image

from magnify.configuration import denorm
from magnify.generator import MagNet



def find_jpg_files(root_folder):
    jpg_files = []
    for root, dirs, files in os.walk(root_folder):
        if len(files) > 0:
            files = natsorted(files)
        for file in files:
            if file.lower().endswith('.jpg'):
                jpg_files.append(os.path.join(root, file))
    return jpg_files


def extract_all_frames(dataset_path, fold_df, i):
    subject = fold_df.iloc[i].subject
    video = fold_df.iloc[i].material
    emote = fold_df.iloc[i].emotion
    dataset = fold_df.iloc[i].dataset.upper()
    onset = str(fold_df.iloc[i]['onset'])
    apex = str(fold_df.iloc[i]['apex'])
    offset = str(fold_df.iloc[i]['offset'])

    # weird naming for 4dmicro
    if dataset == 'FOURD':
        dataset = '4DMicro'
    if dataset == 'CASME3A':
        dataset = 'CASME3'
    if dataset == 'CASME':
        # 'reg_EP01_5-113.jpg'
        onset = 'reg_{}-{}.jpg'.format(video, onset)
        apex = 'reg_{}-{}.jpg'.format(video, apex)
        offset = 'reg_{}-{}.jpg'.format(video, offset)
        images_per_vid = find_jpg_files(os.path.join(dataset_path, dataset, 'TIM16', str(subject), str(video)))
    if dataset == 'CASME2':
        # reg_img46.jpg
        onset = 'reg_img{}.jpg'.format(onset)
        apex = 'reg_img{}.jpg'.format(apex)
        offset = 'reg_img{}.jpg'.format(offset)
        images_per_vid = find_jpg_files(os.path.join(dataset_path, dataset, 'TIM16', str(subject), str(video)))
    if dataset == 'CASME3':
        # 0.jpg
        onset = '{}.jpg'.format(onset)
        apex = '{}.jpg'.format(apex)
        offset = '{}.jpg'.format(offset)                
        images_per_vid = find_jpg_files(os.path.join(dataset_path, dataset, 'TIM16', str(subject), str(video)))
    if dataset == '4DMicro':
        # 0.jpg
        onset = '{}.jpg'.format(onset)
        apex = '{}.jpg'.format(apex)
        offset = '{}.jpg'.format(offset)                   
        images_per_vid = find_jpg_files(os.path.join(dataset_path, dataset, 'TIM16', str(subject), str(video)))
    if dataset == 'MMEW':
        # 0.jpg
        onset = '{}.jpg'.format(onset)
        apex = '{}.jpg'.format(apex)
        offset = '{}.jpg'.format(offset)                   
        images_per_vid = find_jpg_files(os.path.join(dataset_path, dataset, 'TIM16', emote, str(video)))                
    if dataset == 'SAMM':
        # 0.jpg
        onset = '{}.jpg'.format(onset)
        apex = '{}.jpg'.format(apex)
        offset = '{}.jpg'.format(offset)                   
        images_per_vid = find_jpg_files(os.path.join(dataset_path, dataset, 'TIM16', str(subject), str(video)))
    

    return images_per_vid


def extract_onset_apex_offset_frames(dataset_path, fold_df, i):

    subject = fold_df.iloc[i].subject
    video = fold_df.iloc[i].material
    emote = fold_df.iloc[i].emotion
    dataset = fold_df['dataset']
    onset = str(fold_df.iloc[i]['onset'])
    apex = str(fold_df.iloc[i]['apex'])
    offset = str(fold_df.iloc[i]['offset'])
    if dataset == 'CASME':
        # 'reg_EP01_5-113.jpg'
        onset = 'reg_{}-{}.jpg'.format(video, onset)
        apex = 'reg_{}-{}.jpg'.format(video, apex)
        offset = 'reg_{}-{}.jpg'.format(video, offset)
        images_per_vid = find_jpg_files(os.path.join(dataset_path, dataset, 'Cropped', str(subject), str(video)))
    if dataset == 'CASME2':
        # reg_img46.jpg
        onset = 'reg_img{}.jpg'.format(onset)
        apex = 'reg_img{}.jpg'.format(apex)
        offset = 'reg_img{}.jpg'.format(offset)
        images_per_vid = find_jpg_files(os.path.join(dataset_path, dataset, 'Cropped_original', str(subject), str(video)))
    if dataset == 'CASME3':
        # 0.jpg
        onset = '{}.jpg'.format(onset)
        apex = '{}.jpg'.format(apex)
        offset = '{}.jpg'.format(offset)                
        images_per_vid = find_jpg_files(os.path.join(dataset_path, dataset, 'ME_A_cropped', str(subject), str(video)))
    if dataset == '4DMicro':
        # 0.jpg
        onset = '{}.jpg'.format(onset)
        apex = '{}.jpg'.format(apex)
        offset = '{}.jpg'.format(offset)                   
        images_per_vid = find_jpg_files(os.path.join(dataset_path, dataset, 'gray_micro_crop', str(subject), str(video)))
    if dataset == 'MMEW':
        # 0.jpg
        onset = '{}.jpg'.format(onset)
        apex = '{}.jpg'.format(apex)
        offset = '{}.jpg'.format(offset)                   
        images_per_vid = find_jpg_files(os.path.join(dataset_path, dataset, 'Micro_Expression', emote, str(video)))                
    if dataset == 'SAMM':
        # 0.jpg
        onset = '{}.jpg'.format(onset)
        apex = '{}.jpg'.format(apex)
        offset = '{}.jpg'.format(offset)                   
        images_per_vid = find_jpg_files(os.path.join(dataset_path, dataset, 'SAMM_CROP', str(subject), str(video)))
    
    # todo, use df to extract onset apex and offset frames
    try:
        onset_frame = [x for x in images_per_vid if onset in x][0]
        apex_frame = [x for x in images_per_vid if apex in x][0]
        offset_frame = [x for x in images_per_vid if offset in x][0]
    except:
        # cases where naming convention is not consistent in CASME ( 69.jpg and 069.jpg)
        onset = str(fold_df.iloc[i]['onset'])
        apex = str(fold_df.iloc[i]['apex'])
        offset = str(fold_df.iloc[i]['offset'])                
        onset = 'reg_{}-{}.jpg'.format(video, onset.zfill(3))
        apex = 'reg_{}-{}.jpg'.format(video, apex.zfill(3))
        offset = 'reg_{}-{}.jpg'.format(video, offset.zfill(3))                
        onset_frame = [x for x in images_per_vid if onset in x][0]
        apex_frame = [x for x in images_per_vid if apex in x][0]
        offset_frame = [x for x in images_per_vid if offset in x][0]    
    return onset_frame, apex_frame, offset_frame

def extract_frames(dataset_path, fold_df, i):

    subject = fold_df.iloc[i].subject
    video = fold_df.iloc[i].material
    emote = fold_df.iloc[i].emotion
    dataset = fold_df.iloc[i].dataset.upper()
    onset = str(fold_df.iloc[i]['onset'])
    apex = str(fold_df.iloc[i]['apex'])
    offset = str(fold_df.iloc[i]['offset'])

    # weird naming for 4dmicro
    if dataset == 'FOURD':
        dataset = '4DMicro'
    if dataset == 'CASME3A':
        dataset = 'CASME3'


    # switch case for dataset
    if dataset == 'CASME':
        # 'reg_EP01_5-113.jpg'
        # onset = 'reg_{}-{}.jpg'.format(video, onset)
        # apex = 'reg_{}-{}.jpg'.format(video, apex)        
        onset = '{}-{}.jpg'.format(video, onset)
        apex = '{}-{}.jpg'.format(video, apex)            
        images_per_vid = find_jpg_files(os.path.join(dataset_path, dataset, 'Cropped', str(subject), str(video)))
    if dataset == 'CASME2':
        # reg_img46.jpg
        onset = 'reg_img{}.jpg'.format(onset)
        apex = 'reg_img{}.jpg'.format(apex)        
        images_per_vid = find_jpg_files(os.path.join(dataset_path, dataset, 'Cropped', str(subject), str(video)))
    if dataset == 'CASME3':
        # 0.jpg
        onset = '{}.jpg'.format(onset)
        apex = '{}.jpg'.format(apex)        
        images_per_vid = find_jpg_files(os.path.join(dataset_path, dataset, 'ME_A_cropped', str(subject), str(video)))
    if dataset == '4DMicro':
        # 0.jpg     
        onset = '{}.jpg'.format(onset)
        apex = '{}.jpg'.format(apex)        
        images_per_vid = find_jpg_files(os.path.join(dataset_path, dataset, 'Cropped', str(subject), str(video)))
    if dataset == 'MMEW':
        # 0.jpg
        onset = '{}.jpg'.format(onset)
        apex = '{}.jpg'.format(apex)        
        images_per_vid = find_jpg_files(os.path.join(dataset_path, dataset, 'Cropped', emote, str(video)))                
    if dataset == 'SAMM':
        # 0.jpg
        onset = '{}.jpg'.format(onset)
        apex = '{}.jpg'.format(apex)        
        images_per_vid = find_jpg_files(os.path.join(dataset_path, dataset, 'SAMM_CROP', str(subject), str(video)))
    

    try:
        onset_frame = [x for x in images_per_vid if onset in x][0]
        apex_frame = [x for x in images_per_vid if apex in x][0]

    except:
        # cases where naming convention is not consistent in CASME ( 69.jpg and 069.jpg)
        onset = str(fold_df.iloc[i]['onset'])
        apex = str(fold_df.iloc[i]['apex'])

        onset = '{}-{}.jpg'.format(video, onset.zfill(3))
        apex = '{}-{}.jpg'.format(video, apex.zfill(3))
   
        onset_frame = [x for x in images_per_vid if onset in x][0]
        apex_frame = [x for x in images_per_vid if apex in x][0]

    return onset_frame, apex_frame