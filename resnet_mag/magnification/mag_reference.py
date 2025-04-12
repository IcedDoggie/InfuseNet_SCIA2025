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

from magnify.configuration import denorm
from magnify.generator import MagNet



def load_casme_images(num):
    x = np.linspace(2, num, 32)
    model = load_network()
    model.eval()
    root = os.path.join(args.absolute_path, 'datasets/smic_onset_apex')
    subjects = sorted(os.listdir(root))

    for subject in subjects:
        subject_path = os.path.join(root, subject)
        expressions = sorted(os.listdir(subject_path))
        for expression in expressions:
            expression_path = os.path.join(root, subject, expression)
            imgs = sorted(os.listdir(expression_path))
            print(imgs)
            if len(imgs) == 2:
                tt = transforms.Compose([transforms.Resize((384, 384)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

                img1 = Image.open(os.path.join(expression_path, imgs[0]))
                img2 = Image.open(os.path.join(expression_path, imgs[1]))

                img1 = tt(img1)
                img2 = tt(img2)

                for amp_factor in x:
                    mag = torch.from_numpy(np.array([amp_factor])).float()
                    mag = mag.unsqueeze(1).unsqueeze(1).unsqueeze(1)

                    imga = img1.view(1, img1.shape[0], img1.shape[1], img1.shape[2]).cuda()
                    imgb = img2.view(1, img2.shape[0], img2.shape[1], img2.shape[2]).cuda()
                    mag = mag.cuda()

                    y_hat, _, _ = model(imga, imgb, mag)
                    y_hat = denorm(y_hat)
                    demo = os.path.join("./datasets/casme2", "casme_magnified_" + str(num))
                    demo_dir = os.path.join(demo, subject, expression)
                    if not os.path.exists(demo_dir):
                        os.makedirs(demo_dir)
                    print(demo_dir)
                    amp_factor = round(amp_factor, 2)
                    save_image(y_hat.data, demo_dir + '/' + str(amp_factor) + '.jpg')


def load_test_image(num):
    x = np.linspace(2,num,32)
    model = load_network()
    model.eval()
    root = '/home/hq/Documents/WorkingRepos/resnet_mag/magnification/datasets/smic_onset_apex/'
    imgs = sorted(os.listdir(root))
    
    if len(imgs) == 2:
        tt = transforms.Compose([transforms.Resize((384, 384)),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        img1 = Image.open(root+imgs[0])
        img2 = Image.open(root+imgs[1])
        img1, img2 = tt(img1), tt(img2)
        for amp_factor in x:
            mag = torch.from_numpy(np.array([amp_factor])).float()
            mag = mag.unsqueeze(1).unsqueeze(1).unsqueeze(1)

            imga = img1.view(1, img1.shape[0], img1.shape[1], img1.shape[2]).cuda()
            imgb = img2.view(1, img2.shape[0], img2.shape[1], img2.shape[2]).cuda()
            mag = mag.cuda()
            model = model.cuda()

            y_hat, _, _ = model(imga, imgb, mag)
            y_hat = denorm(y_hat)
            y_hat = y_hat

            amp_factor = round(amp_factor, 2)

            y_hat = y_hat[0]
            y_hat = y_hat.permute(1, 2, 0)
            y_hat = y_hat.cpu().detach().numpy()


            # y_hat = y_hat.detach().numpy()
            plt.imshow(y_hat)
            plt.show()

            a = 'finished'

            
            
            # save_image(y_hat.data, '/home/hq/Documents/WorkingRepos/resnet_mag/magnification/datasets/own_attempt/'+ str(amp_factor) + '.jpg')
