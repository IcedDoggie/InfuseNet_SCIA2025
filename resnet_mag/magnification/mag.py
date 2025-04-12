import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from tqdm import tqdm

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader


from magnify.configuration import denorm
from magnify.generator import MagNet

from mag_model import load_network
from mag_utils import extract_frames
from mag_utils import extract_all_frames

def magnification_process_sequence(images_name_list):
    transform = transforms.Compose([transforms.Resize((384, 384)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  
    
    # TODO: 1. write a loop to loop through all tim frames 
    # 2. magnify each frame
    # 3. saves as output
    # repeat for all mag factor
    amp_factor_list = np.arange(start=1, stop=10, step=1)
    # amp_factor = 1
    for amp_factor in amp_factor_list: 
        for i in range(len(images_name_list)):
            if len(images_name_list) - i == 1:
                # if it is the last image, then take current and prev frame
                reference_frame_name = images_name_list[i - 1]
                current_frame_name = images_name_list[i]
                img_name = current_frame_name.split('/')[-1]
            else:
                # if it is NOT the last time, then take current and next frame
                current_frame_name = images_name_list[i]
                reference_frame_name = images_name_list[i + 1]     
                img_name = current_frame_name.split('/')[-1]           

            # preparing a new folder to save magnified files
            front_part_of_folder = '/'.join(current_frame_name.split('/')[0:6])
            back_part_of_folder = '/'.join(current_frame_name.split('/')[7:-1])
            # magnified_folder_name = 'mag1_TIM16'
            magnified_folder_name = 'mag{}_TIM16'.format(str(amp_factor))
            folder_path = os.path.join(front_part_of_folder, magnified_folder_name, back_part_of_folder)
            filename = os.path.join(folder_path, img_name)
            if os.path.exists(folder_path) == False:
                os.makedirs(folder_path)            

            img_a = Image.open(current_frame_name)
            img_b = Image.open(reference_frame_name)

            if img_a.mode == 'L':
                # Convert the grayscale image to RGB
                img_a = img_a.convert('RGB')
            if img_b.mode == 'L':
                # Convert the grayscale image to RGB
                img_b = img_b.convert('RGB')        

            img_a, img_b = transform(img_a), transform(img_b)
            img_a = img_a.view(1, img_a.shape[0], img_a.shape[1], img_a.shape[2]).cuda()
            img_b = img_b.view(1, img_b.shape[0], img_b.shape[1], img_b.shape[2]).cuda()        

            # magnification process (magnify from n--N, n=1, N=9)
            # x = np.arange(start=1, stop=10, step=1)
            mag = torch.from_numpy(np.array([amp_factor])).float()
            mag = mag.unsqueeze(1).unsqueeze(1).unsqueeze(1)        
            mag = mag.cuda() 
            y_hat, _, _ = mag_model(img_a, img_b, mag)
            y_hat = denorm(y_hat)    
            amp_factor = round(amp_factor, 2)
            filename_with_amp = filename.split('.jpg')[0] + '_' + str(amp_factor) + '.jpg'
            save_image(y_hat.data, filename_with_amp)


def magnification_process(images_name_list):
    transform = transforms.Compose([transforms.Resize((384, 384)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    
    
    # for i in range(len(images_name_list)):
    #     if len(images_name_list) - i == 1:
    #         # if it is the last image, then take current and prev frame
    #         reference_frame_name = images_name_list[i - 1]
    #         current_frame_name = images_name_list[i]
    #         img_name = current_frame_name.split('/')[-1]
    #     else:
    #         # if it is NOT the last time, then take current and next frame
    #         current_frame_name = images_name_list[i]
    #         reference_frame_name = images_name_list[i + 1]     
    #         img_name = current_frame_name.split('/')[-1]   

    current_frame_name = images_name_list[0]
    reference_frame_name = images_name_list[1]
    img_name = current_frame_name.split('/')[-1]

    # preparing a new folder to save magnified files
    front_part_of_folder = '/'.join(current_frame_name.split('/')[0:6])
    back_part_of_folder = '/'.join(current_frame_name.split('/')[7:-1])
    magnified_folder_name = 'mag1_TIM16'
    folder_path = os.path.join(front_part_of_folder, magnified_folder_name, back_part_of_folder)
    filename = os.path.join(folder_path, img_name)
    if os.path.exists(folder_path) == False:
        os.makedirs(folder_path)            

    img_a = Image.open(current_frame_name)
    img_b = Image.open(reference_frame_name)

    if img_a.mode == 'L':
        # Convert the grayscale image to RGB
        img_a = img_a.convert('RGB')
    if img_b.mode == 'L':
        # Convert the grayscale image to RGB
        img_b = img_b.convert('RGB')        

    img_a, img_b = transform(img_a), transform(img_b)
    img_a = img_a.view(1, img_a.shape[0], img_a.shape[1], img_a.shape[2]).cuda()
    img_b = img_b.view(1, img_b.shape[0], img_b.shape[1], img_b.shape[2]).cuda()        

    # magnification process (magnify from n--N, n=1, N=9)
    # x = np.arange(start=1, stop=10, step=1)
    x = 1
    mag = torch.from_numpy(np.array([amp_factor])).float()
    mag = mag.unsqueeze(1).unsqueeze(1).unsqueeze(1)        
    mag = mag.cuda() 
    y_hat, _, _ = mag_model(img_a, img_b, mag)
    y_hat = denorm(y_hat)    
    amp_factor = round(amp_factor, 2)
    filename_with_amp = filename.split('.jpg')[0] + '_' + str(amp_factor) + '.jpg'
    save_image(y_hat.data, filename_with_amp)

    # magnification process (normal distribution of magnified onset to apex, from 2 to num)
    # num = 10
    # x = np.linspace(2, num, 32)

    # for amp_factor in x:
    #     mag = torch.from_numpy(np.array([amp_factor])).float()
    #     mag = mag.unsqueeze(1).unsqueeze(1).unsqueeze(1)        
    #     mag = mag.cuda() 
    #     y_hat, _, _ = mag_model(img_a, img_b, mag)
    #     y_hat = denorm(y_hat)    
    #     amp_factor = round(amp_factor, 2)
    #     filename_with_amp = filename.split('.jpg')[0] + '_' + str(amp_factor) + '.jpg'
    #     save_image(y_hat.data, filename_with_amp)

    # # magnification process (only for two frames.)
    # amp_factor = 5
    # mag = torch.from_numpy(np.array([amp_factor])).float()
    # mag = mag.unsqueeze(1).unsqueeze(1).unsqueeze(1)        
    # mag = mag.cuda()     
    # y_hat, _, _ = mag_model(img_a, img_b, mag)
    # y_hat = denorm(y_hat)    
    # amp_factor = round(amp_factor, 2)
    # filename_with_amp = filename.split('.jpg')[0] + '_' + str(amp_factor) + '.jpg'
    # save_image(y_hat.data, filename_with_amp)


if __name__ == '__main__':

    # load model (input to model: )
    checkpoint_path = '/home/hq/Documents/Weights/Magnification/generator_212000.pth'
    mag_model = load_network(checkpoint_path)
    mag_model = mag_model.cuda()
    mag_model.eval()
    a = 1

    # load data
    df = pd.read_csv('/home/hq/Documents/WorkingRepos/MER_BASELINE_CD6ME/metadata_csv/cross_dataset_seq.csv')
    dataset_path = '/home/hq/Documents/data'

    df.loc[df['dataset'] == ('casme'),'subject'] = 'sub' + df['subject'].loc[df['dataset'] == ('casme')] # add sub to df for casme
    df.loc[df['dataset'] == ('casme2'),'subject'] = 'sub' + df['subject'].loc[df['dataset'] == ('casme2')] # add sub to df for casme2
    
    # df = df.loc[(df['dataset'] != 'casme') & (df['dataset'] != 'casme2')]
    # df = df.loc[(df['dataset'] == 'casme3a') | (df['dataset'] == 'mmew') ]
    # df = df.loc[(df['dataset'] == 'fourd')]
    for dataset in df['dataset'].unique():
        dataset_df = df.loc[df['dataset'] == dataset]
        print(dataset)
        # # magnification process (normal distribution of magnified onset to apex, from 2 to num)
        # for i in tqdm(range(len(df))):
        #     frames = extract_frames(dataset_path, df, i)
        #     magnification_process(images_name_list=frames)
        
        # # magnification process (only for two frames.)
        # for i in tqdm(range(len(dataset_df))):
        #     # if 
        #     frames = extract_frames(dataset_path, dataset_df, i)
        #     magnification_process(images_name_list=frames)            

        # magnification process (for whole sequence)
        for i in tqdm(range(len(dataset_df))):
            frames = extract_all_frames(dataset_path, dataset_df, i)
            magnification_process_sequence(images_name_list=frames)  
            
        



   




    # save the images in npy
    





