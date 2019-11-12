import os
import pandas as pd
import torch

from transforms import transforms
from utils.autoaugment import ImageNetPolicy

# pretrained model checkpoints
pretrained_model = {'resnet50' : './models/pretrained/resnet50-19c8e357.pth',
                    'senet154' : './models/pretrained/senet154-c7b49a05.pth',
                    'se_resnext101_32x4d' : './models/pretrained/se_resnext101_32x4d-3b2fe3d8.pth',
                    'se_resnet50' : './models/pretrained/se_resnet50-ce0d4300.pth',
                    }

# transforms dict
def load_data_transformers(resize_reso=512, crop_reso=448, swap_num=[7, 7]):
    center_resize = 600
    Normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    data_transforms = {
       	'swap': transforms.Compose([
            transforms.Randomswap((swap_num[0], swap_num[1])),
        ]),
        'common_aug': transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomCrop((crop_reso,crop_reso)),
            transforms.RandomHorizontalFlip(),
        ]),
        'train_totensor': transforms.Compose([
            transforms.Resize((crop_reso, crop_reso)),
            # ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val_totensor': transforms.Compose([
            transforms.Resize((crop_reso, crop_reso)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'test_totensor': transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.CenterCrop((crop_reso, crop_reso)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'None': None,
    }
    return data_transforms


class LoadConfig(object):
    def __init__(self, args, version):
        if version == 'train':
            get_list = ['train', 'val']
        elif version == 'val':
            get_list = ['val']
        elif version == 'test':
            get_list = ['test']
        else:
            raise Exception("train/val/test ???\n")

        ###############################
        #### add dataset info here ####
        ###############################

        # put image data in $PATH/data
        # put annotation txt file in $PATH/anno

        if args.dataset == 'MTFood-1000':
            self.dataset = args.dataset
            self.rawdata_root = './dataset/MTFood-1000/data/train'
            self.anno_root = './dataset/MTFood-1000/anno'
            self.numcls = 1000
        elif args.dataset == 'CUB':
            self.dataset = args.dataset
            self.rawdata_root = './dataset/CUB_200_2011/data'
            self.anno_root = './dataset/CUB_200_2011/anno'
            self.numcls = 200
        elif args.dataset == 'STCAR':
            self.dataset = args.dataset
            self.rawdata_root = './dataset/st_car/data'
            self.anno_root = './dataset/st_car/anno'
            self.numcls = 196
        elif args.dataset == 'AIR':
            self.dataset = args.dataset
            self.rawdata_root = './dataset/aircraft/data'
            self.anno_root = './dataset/aircraft/anno'
            self.numcls = 100
        else:
            raise Exception('dataset not defined ???')

        # annotation file organized as :
        # path/image_name cls_num\n

        if 'train' in get_list:
             self.train_anno = pd.read_csv(os.path.join(self.anno_root, 'train_label.txt'),\
                                           sep=" ",\
                                           header=None,\
                                           names=['ImageName', 'label'])

        if 'val' in get_list:
            self.val_anno = pd.read_csv(os.path.join(self.anno_root, 'val_label.txt'),\
                                           sep=" ",\
                                           header=None,\
                                           names=['ImageName', 'label'])

        if 'test' in get_list:
            self.test_anno = pd.read_csv(os.path.join(self.anno_root, 'test_label.txt'),\
                                           sep=" ",\
                                           header=None,\
                                           names=['ImageName', 'label'])

        self.swap_num = args.swap_num

        self.save_dir = './net_model'
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.backbone = args.backbone

        self.use_dcl = True
        self.use_backbone = False if self.use_dcl else True
        self.use_Asoftmax = False
        self.use_focal_loss = False
        self.use_fpn = False
        self.use_hier = False

        self.weighted_sample = False
        self.cls_2 = False
        self.cls_2xmul = True
        self.use_adam = args.use_adam

        self.log_folder = './logs'
        if not os.path.exists(self.log_folder):
            os.mkdir(self.log_folder)




