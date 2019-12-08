"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
#coding=utf-8
import os
import json
import csv
import argparse
import pandas as pd
import numpy as np
from math import ceil
from tqdm import tqdm
import pickle
import shutil
import xlwt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torchvision import datasets, models
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from transforms import transforms
from models.LoadModel import MainModel
from config import LoadConfig, load_data_transformers

from PIL import Image
from glob import glob
from tqdm import tqdm

from misc_functions import get_example_params, save_class_activation_images

def parse_args():
    parser = argparse.ArgumentParser(description='dcl parameters')
    parser.add_argument('--data', dest='dataset',
                        default='MTFood-1000', type=str)
    parser.add_argument('--backbone', dest='backbone',
                        default='resnet50', type=str)
    parser.add_argument('--b', dest='batch_size',
                        default=25, type=int)
    parser.add_argument('--nw', dest='num_workers',
                        default=32, type=int)
    parser.add_argument('--ver', dest='version',
                        default='val', type=str)
    parser.add_argument('--save', dest='resume',
                        default=None, type=str)
    parser.add_argument('--size', dest='resize_resolution',
                        default=512, type=int)
    parser.add_argument('--crop', dest='crop_resolution',
                        default=448, type=int)
    parser.add_argument('--ss', dest='save_suffix',
                        default=None, type=str)
    parser.add_argument('--acc_report', dest='acc_report',
                        action='store_true')
    parser.add_argument('--swap_num', default=[7, 7],
                    nargs=2, metavar=('swap1', 'swap2'),
                    type=int, help='specify a range')
    parser.add_argument('--use-adam', action='store_true')
    args = parser.parse_args()
    return args

class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        x = self.model.model(x)
        x.register_hook(self.save_gradient)
        conv_output = x
        x = self.model.avgpool(x)
        # for module_pos, module in self.model.features._modules.items():
        #     x = module(x)  # Forward
        #     if int(module_pos) == self.target_layer:
        #         x.register_hook(self.save_gradient)
        #         conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.classifier(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        pred_class = model_output.data.numpy()[0].argsort()[-3:][::-1]
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.model.zero_grad()
        self.model.classifier.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.
        return cam, pred_class

def pil_loader(imgpath):
    with open(imgpath, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


if __name__ == '__main__':
    # Get params
    # target_example = 0  # Snake
    # (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
    #     get_example_params(target_example)
    # # Grad cam
    # grad_cam = GradCam(pretrained_model)
    # # Generate cam mask
    # cam = grad_cam.generate_cam(prep_img, target_class)
    # # Save mask
    # save_class_activation_images(original_image, cam, file_name_to_export)
    # print('Grad cam completed')

    args = parse_args()
    Config = LoadConfig(args, args.version)
    Config.cls_2 = True
    Config.cls_2xmul = False

    models = [
        'wide_resnet50_2',
        'resnet50',
        'resnext50_32x4d',
        'se_resnext101_32x4d'
    ]
    weights = {
        'resnet50': 'net_model/resnet50/weights_65_109_0.7044_0.8736.pth',
        'resnext50_32x4d': 'net_model/resnext50_32x4d/weights_59_1549_0.7046_0.8772.pth',
        'se_resnext101_32x4d': 'net_model/se_resnext101_32x4d/weights_18_4583_0.7152_0.8783.pth',
        'wide_resnet50_2': 'net_model/wide_resnet50_2/weights_58_4051_0.7255_0.8865.pth'
    }
    imgs = glob('good_case_2/*.jpg')
    crop_reso = 448
    transformer = transforms.Compose([
            transforms.Resize((crop_reso, crop_reso)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    for model_name in models:
        print(model_name)
        Config.backbone = model_name
        model = MainModel(Config)
        model_dict=model.state_dict()
        pretrained_dict=torch.load(weights[model_name])
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.eval()

        grad_cam = GradCam(model)
        save_path = os.path.join('./good_case_2/{}/'.format(model_name))
        for img in tqdm(imgs):
            original_image = pil_loader(img)
            img_name = img.split('/')[-1].split('.')[0]
            target_class = int(img_name.split('_')[0])
            img_tensor = transformer(original_image).unsqueeze(0)
            cam, pred_class = grad_cam.generate_cam(img_tensor, target_class)
            resize_img = original_image.resize((448, 448))
            string_pred_class = '-'.join([str(pred_class[i]) for i in range(3)])
            save_class_activation_images(resize_img, cam, img_name + '_pred{}'.format(pred_class), save_path)
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')