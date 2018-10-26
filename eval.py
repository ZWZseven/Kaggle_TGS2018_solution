import os
import numpy as np # linear algebra
import pandas as pd # data processing, 
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

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

def validation(model: nn.Module, criterion, valid_loader):
    print("Validation on hold-out....")
    model.eval()
    losses = []
    jaccard = []
    ious = []
    for inputs, targets in valid_loader:
        inputs = variable(inputs, volatile=True)
        targets = variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.data[0])
        jaccard += [get_jaccard(targets, (outputs > 0).float()).data[0]]
        ious += [get_iou_vector(targets, (outputs > 0).float())]
    
    valid_ious = np.mean(ious)
    
    valid_loss = np.mean(losses)  # type: float

    valid_jaccard = np.mean(jaccard)

    print('Valid loss: {:.5f}, jaccard: {:.5f}, ious: {:.5f}'.format(valid_loss, valid_jaccard, valid_ious))
    metrics = {'valid_loss': valid_loss, 'jaccard_loss': valid_jaccard, 'ious_loss': valid_ious}
    return metrics
def validation4(model: nn.Module, criterion, valid_loader):
    print("Validation on hold-out....")
    model.eval()
    losses = []
    jaccard = []
    ious = []
    for inputs, targets in valid_loader:
        inputs = variable(inputs, volatile=True)
        targets = variable(targets)
        outputs,o2,o3,o4= model(inputs)
        loss = criterion(outputs,o2,o3,o4, targets)
        losses.append(loss.data[0])
        jaccard += [get_jaccard(targets, (outputs > 0).float()).data[0]]
        ious += [get_iou_vector(targets, (outputs > 0).float())]
    
    valid_ious = np.mean(ious)
    
    valid_loss = np.mean(losses)  # type: float

    valid_jaccard = np.mean(jaccard)

    print('Valid loss: {:.5f}, jaccard: {:.5f}, ious: {:.5f}'.format(valid_loss, valid_jaccard, valid_ious))
    metrics = {'valid_loss': valid_loss, 'jaccard_loss': valid_jaccard, 'ious_loss': valid_ious}
    return metrics
def validation5(model: nn.Module, criterion, valid_loader):
    print("Validation on hold-out....")
    model.eval()
    losses = []
    jaccard = []
    ious = []
    for inputs, targets in valid_loader:
        inputs = variable(inputs, volatile=True)
        targets = variable(targets)
        outputs,o2,o3,o4,o5= model(inputs)
        loss = criterion(outputs,o2,o3,o4,o5, targets)
        losses.append(loss.data[0])
        jaccard += [get_jaccard(targets, (outputs > 0).float()).data[0]]
        ious += [get_iou_vector(targets, (outputs > 0).float())]
    
    valid_ious = np.mean(ious)
    
    valid_loss = np.mean(losses)  # type: float

    valid_jaccard = np.mean(jaccard)

    print('Valid loss: {:.5f}, jaccard: {:.5f}, ious: {:.5f}'.format(valid_loss, valid_jaccard, valid_ious))
    metrics = {'valid_loss': valid_loss, 'jaccard_loss': valid_jaccard, 'ious_loss': valid_ious}
    return metrics
def get_jaccard(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1).sum(dim = -1)
    union = y_true.sum(dim=-2).sum(dim=-1).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1).sum(dim = -1)

    return (intersection / (union - intersection + epsilon)).mean()


import numpy as np # linear algebra

def get_iou_vector_single(A, B):
    #A:True,B:Pred
    #print(A.shape)
    batch_size = A.shape[0]
    print(batch_size)
    #metric = []
    #for batch in range(batch_size):
    t, p = A.squeeze(), B.squeeze()
    if np.count_nonzero(t) == 0 and np.count_nonzero(p) > 0:
        metric=0
        return np.mean(metric)
        #continue
    if np.count_nonzero(t) == 0 and np.count_nonzero(p) == 0:
        metric=1
        return np.mean(metric)
        #continue

    iou = jaccard(t, p)
    #print(iou)
    thresholds = np.arange(0.5, 1, 0.05)
    s = []
    for thresh in thresholds:
        #s.append(iou > thresh)
        s.append(iou > thresh)
    metric=np.mean(s)
    #print(metric.shape)
    return np.mean(metric)

def get_iou_vector(A, B):
    #A:True,B:Pred
    #print(A.shape)
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch].squeeze(), B[batch].squeeze()
        if np.count_nonzero(t) == 0 and np.count_nonzero(p) > 0:
            metric.append(0)
            continue
        if np.count_nonzero(t) == 0 and np.count_nonzero(p) == 0:
            metric.append(1)
            continue

        iou = jaccard(t, p)
        #print(iou)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            #s.append(iou > thresh)
            s.append(iou > thresh)
        metric.append(np.mean(s))
    #print(metric.shape)
    return np.mean(metric)



def jaccard(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum().sum()
    union = y_true.sum().sum() + y_pred.sum().sum()

    return ((intersection + epsilon)/ (union - intersection + epsilon)).mean()

#thresholds = np.linspace(0, 1, 50)
#ious = np.array([get_iou_vector(y_valid_ori, np.int32(preds_valid > threshold)) for threshold in tqdm_notebook(thresholds)])


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

#json_1 = {'num':1112, 'date':datetime.now()}
#print(json.dumps(json_1, cls=MyEncoder))

        
# sume helper functions
def variable(x, volatile=False):
    if isinstance(x, (list, tuple)):
        return [variable(y, volatile=volatile) for y in x]
    return cuda(Variable(x, volatile=volatile))

def cuda(x):
    return x.cuda(async=True) if torch.cuda.is_available() else x

def write_event(log, lr, step: int, **data):
    data['lr'] = lr
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True, cls=MyEncoder))
    log.write('\n')
    log.flush()

###########################################################
# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in
    
    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)
    
