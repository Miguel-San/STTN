import os
import cv2
import io
import glob
import scipy
import json
import zipfile
import random
import collections
import torch
import math
import numpy as np
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageFilter
from skimage.color import rgb2gray, gray2rgb
from core.utils import ZipReader, create_random_shape_with_random_motion
from core.utils import Stack, ToTorchFormatTensor, GroupRandomHorizontalFlip


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args: dict, split='train', debug=False):
        self.args = args
        self.split = split
        self.sample_length = args['sample_length']
        self.stride = args["stride"]
        self.size = self.w, self.h = (args['w'], args['h'])
        self.img_ext = args['img_ext']

        with open(os.path.join(args['data_root'], args['name'], split+'.json'), 'r') as f:
            self.video_dict = json.load(f)
        self.video_names = list(self.video_dict.keys())
        if debug or split != 'train':
            self.video_names = self.video_names[:100]

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(), ])

    def __len__(self):
        # return len(self.video_names)
        video_name = self.video_names[0]
        return self.video_dict[video_name]-(self.sample_length-1)-(self.stride-1)*(self.sample_length-1)
    
    def video_length(self):
        return self.video_dict[self.video_names[0]]

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('Loading error in video {}'.format(self.video_names[0]))
            item = self.load_item(0)
        return item

    def load_item(self, index):
        video_name = self.video_names[0]
        # all_frames = [f"{str(i).zfill(5)}."+self.img_ext for i in range(self.video_dict[video_name])]
        # all_masks = create_random_shape_with_random_motion(
        #     len(all_frames), imageHeight=self.h, imageWidth=self.w)
        all_masks = []
        for fname in os.listdir('{}/{}/masks'.format(self.args['data_root'], self.args['name'])):
            mask = Image.fromarray(cv2.imread('{}/{}/masks'.format(
                self.args['data_root'], self.args['name']) + '/' + fname)).convert("L")
            mask = mask.resize(self.size)
            all_masks.append(mask)

        ref_index = get_ref_index(self.video_length(), self.sample_length, self.stride, pivot=index)
        frame_img_name = [f"{str(i).zfill(5)}."+self.img_ext for i in ref_index]
        # read video frames
        frames = []
        masks = []
        for i, idx in enumerate(ref_index):
            # img = ZipReader.imread('{}/{}/JPEGImages/{}.zip'.format(
            #     self.args['data_root'], self.args['name'], video_name), all_frames[idx]).convert('RGB')
            img = Image.fromarray(cv2.imread('{}/{}/JPEGImages/{}'.format(
                self.args['data_root'], self.args['name'], video_name) + '/' + frame_img_name[i])).convert("L")
            img = img.resize(self.size)
            frames.append(img)
            # masks.append(all_masks[idx])
            masks.append(all_masks[np.random.randint(len(all_masks))])
        if self.split == 'train':
            pass
            # frames = GroupRandomHorizontalFlip()(frames)
        # To tensors
        frame_tensors = self._to_tensors(frames)*2.0 - 1.0
        mask_tensors = self._to_tensors(masks)
        return frame_tensors, mask_tensors


def get_ref_index(length, sample_length, stride, pivot=None):
    # if random.uniform(0, 1) > 0.5:
    #     ref_index = random.sample(range(length), sample_length)
    #     ref_index.sort()
    # else:

    #  Works for no stride
    # pivot = random.randint(0, length-sample_length)
    # ref_index = [pivot+i for i in range(sample_length)]
    if pivot is None:
        pivot = random.randint(0, length-sample_length-(stride-1)*(sample_length-1))
    ref_index = [pivot+i*stride for i in range(sample_length)]
    return ref_index


# class TestDataset(Dataset):
#     def load_item(self, index):
#         video_name = self.video_names[index]
#         all_frames = [f"{str(i).zfill(5)}."+self.img_ext for i in range(self.video_dict[video_name])]
#         all_masks = []
#         for fname in os.listdir('{}/{}/masks'.format(self.args['data_root'], self.args['name'])):
#             mask = Image.fromarray(cv2.imread('{}/{}/masks'.format(
#                 self.args['data_root'], self.args['name']) + '/' + fname)).convert("L")
#             mask = mask.resize(self.size)
#             all_masks.append(mask)

#         ref_index = get_ref_index(len(all_frames), self.sample_length, self.stride)
#         frames = []
#         masks = []
#         for i in range(len(all_frames)):
#             img = Image.fromarray(cv2.imread('{}/{}/JPEGImages/{}'.format(
#                 self.args['data_root'], self.args['name'], video_name) + '/' + all_frames[i])).convert("L")
#             img = img.resize(self.size)
#             frames.append(img)

#             masks.append(all_masks[np.random.randint(len(all_masks))])

#         frame_tensors = self._to_tensors(frames)*2.0 - 1.0
#         mask_tensors = self._to_tensors(masks)
#         return frame_tensors, mask_tensors
    
#     @classmethod
#     def get_win_idx_from_frame(cls, idx, sample_length, nframes):
#         first_win_idx = idx // (sample_length-1)
#         last_win_idx = idx

#         win_indices = []
#         frame_indices = []
#         k = sample_length - (last_win_idx - first_win_idx) - 1
#         for w_idx in range(first_win_idx, last_win_idx+1):
#             if w_idx >= nframes-sample_length:
#                 continue

#             f_idx = sample_length - k - 1   
#             k+=1
#             if f_idx < 0 or f_idx >= sample_length:
#                 continue

#             win_indices.append(w_idx)
#             frame_indices.append(f_idx)    

#         return win_indices, frame_indices