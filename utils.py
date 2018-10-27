import os
import numpy as np # 
import pandas as pd # 
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
#from skimage.util.montage import montage2d as montage
import cv2
import random
from datetime import datetime
import json
import gc

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torch.backends.cudnn as cudnn
import torch.backends.cudnn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.transforms import ToTensor, Normalize, Compose

from sklearn.model_selection import train_test_split

from tqdm import tqdm
from pathlib import Path

from skimage.morphology import label
from skimage.transform import resize

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def multi_rle_encode(img):
    labels = label(img)
    return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]

def rle_decode(mask_rle, shape=(101, 101)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction
def upsample(img,img_size):
    return cv2.resize(img.squeeze(), (img_size, img_size), interpolation=cv2.INTER_LINEAR)

def upsamplearray(arr,img_size):
    out=[upsample(x.squeeze(),img_size) for x in arr]
    return np.array(out)    

def downsample(img,img_size):
    #img_size=101
    return cv2.resize(img.squeeze(), (img_size, img_size), interpolation=cv2.INTER_LINEAR)

img_size_ori = 101
img_size_target = 128

def downsamplearray(arr):
    out=[downsample(x.squeeze(),101) for x in arr]
    return np.array(out).reshape(-1,101,101,1)

def restore(img):
    img = downsample(img,128)
    return img.squeeze()[13:114,13:114]

def flpadding(img,a,b):#13,14
    m,n=img.shape
    output=np.zeros([m+a+b,n+a+b])#([127,127])
    output[a:m+a,a:n+a]=img.squeeze()
    imglr=np.fliplr(img.squeeze())
    output[a:m+a,0:a]=imglr[:,-a:]
    #output[13:114,-13:]=imglr[:,0:13]
    output[a:m+a,-b:]=imglr[:,0:b]
    imgud1=np.flipud(output)
    output[0:a,:]=imgud1[-2*a:-a,:]
    output[-b:,:]=imgud1[b:2*b,:]  
    return output

def imgexpand0(img):
    t_size=128
    return upsample(flpadding(img.squeeze(),13,14),t_size).reshape(t_size,t_size,1)
    
def imgexarray0(arr):
    out=[imgexpand0(x) for x in arr]
    return np.array(out)

def imgexpand03(img):
    t_size=128#224
    output0=np.zeros([t_size,t_size,3])#([192,192,3])#([127,127])
    output=imgexpand0(img.squeeze())#np.zeros([128,128])#
    for i in range(3):
        output0[:,:,i]=output.squeeze()
    return output0

def imgexarray03(arr):
    out=[imgexpand03(x) for x in arr]
    return np.array(out)

def imgexpand033(img):
    t_size=128#224
    output0=np.zeros([t_size,t_size,3])#([192,192,3])#([127,127])
    output=imgexpand0(img[:,:,0].squeeze())#np.zeros([128,128])#
    for i in range(3):
        output0[:,:,i]=output.squeeze()
    return output0


def masks_as_image(in_mask_list):
    all_masks = np.zeros((101, 101), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)

def mask_overlay(image, mask, color=(0, 1, 0)):
    """
    Helper function to visualize mask on the top of the image
    """
    mask = np.dstack((mask, mask, mask)) * np.array(color)
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0
    img[ind] = weighted_sum[ind]    
    return img

def imshow(img, mask, title=None):
    """Imshow for Tensor."""
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    mask = mask.numpy().transpose((1, 2, 0))
    mask = np.clip(mask, 0, 1)
    fig = plt.figure(figsize = (6,6))
    plt.imshow(mask_overlay(img, mask))
    if title is not None:
        plt.title(title)
    plt.pause(0.001) 



# Source https://www.kaggle.com/bguberfain/unet-with-depth
def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

