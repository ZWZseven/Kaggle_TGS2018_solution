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

import torch.backends.cudnn as cudnn
import torch.backends.cudnn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.transforms import ToTensor, Normalize, Compose

from tqdm import tqdm

from skimage.morphology import label
from skimage.transform import resize



# 'Cyclical Learning Rates for Training Neural Networks'- Leslie N. Smith, arxiv 2017
#       https://arxiv.org/abs/1506.01186
#       https://github.com/bckenstler/CLR

class CyclicLR():

    def __init__(self, base_lr=0.001, max_lr=0.006, step=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step = step
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: (0.5)**(x-1)
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step != None:
            self.step = new_step
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step))
        x = np.abs(self.clr_iterations/self.step - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)

    def get_rate(self, epoch=None, num_epoches=None):

        self.trn_iterations += 1
        self.clr_iterations += 1
        lr = self.clr()

        return lr

    def __str__(self):
        string = 'Cyclical Learning Rates\n' \
                + 'base_lr=%0.3f, max_lr=%0.3f'%(self.base_lr, self.max_lr)
        return string



# net ------------------------------------
# https://github.com/pytorch/examples/blob/master/imagenet/main.py ###############
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]

    assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]

    assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr


def save_checkpoint(state, is_best, filename='model_1.pt'):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print ("=> Saving a new best")
        torch.save(state, filename) # save checkpoint
    else:
        print ("=> Validation Accuracy did not improve")

"""# Training the Model
for epoch in range(num_epochs):
 train(...) # Train
 acc = eval(...) # Evaluate after every epoch

# Some stuff with acc(accuracy)
 ...

# Get bool not ByteTensor
 is_best = bool(acc.numpy() > best_accuracy.numpy())
 # Get greater Tensor to keep track best acc
 best_accuracy = torch.FloatTensor(max(acc.numpy(), best_accuracy.numpy()))
 # Save checkpoint if is a new best
 save_checkpoint({
 'epoch': start_epoch + epoch + 1,
 'state_dict': model.state_dict(),
 'best_accuracy': best_accuracy
 }, is_best,model_path)
"""

# main train routine
# Implementation from  https://github.com/ternaus/robot-surgery-segmentation
def train(lr, model, criterion, train_loader, valid_loader, validation, init_optimizer, n_epochs=1, fold=1):
    optimizer = init_optimizer(lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5,patience=20)################
    #model = nn.DataParallel(model, device_ids=None)
    if torch.cuda.is_available():
        model.cuda()
    isrestore = False   
    startepoch = 0
    model_path = Path('model_{fold}.pt'.format(fold=fold))
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
        isrestore = True
        startepoch = epoch
        
        valid_metrics = validation(model, criterion, valid_loader)
        valid_loss = valid_metrics['valid_loss']
        ious_loss = valid_metrics['ious_loss']
        print(ious_loss)
        best_accuracy=-ious_loss
    else:
        epoch = 1
        step = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(model_path))
    

    report_each = 50
    log = open('train_{fold}.log'.format(fold=fold),'at', encoding='utf8')
    valid_losses = []
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        random.seed()
        tq = tqdm(total=len(train_loader) *  BATCH_SIZE)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        tl = train_loader
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                inputs, targets = variable(inputs), variable(targets)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss.data[0])
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                if i and i % report_each == 0:
                    write_event(log, lr, step, loss=mean_loss)
            write_event(log, lr, step, loss=mean_loss)########################lr
            tq.close()
            #save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader)
            write_event(log, lr, step, **valid_metrics)#############################
            valid_loss = valid_metrics['valid_loss']

            acc= -valid_metrics['ious_loss']
            scheduler.step(acc)###########################
            lr=get_learning_rate(optimizer)###############################
            
            if epoch==1:
                best_accuracy=-valid_metrics['ious_loss']#valid_loss#(valid_loss.from_numpy())
                is_best=True
                if is_best:
                    print ("=> Saving a new best")
                    save(epoch) # save checkpoint
                else:
                    print ("=> Validation Accuracy did not improve")
            else:
                #if isrestore == False:
                is_best = bool(acc < best_accuracy)#(acc.numpy() > best_accuracy.numpy())
                best_accuracy = min(acc, best_accuracy)#torch.FloatTensor(min(acc.numpy(), best_accuracy.numpy()))
                if is_best:
                    print ("=> Saving a new best")
                    save(epoch) # save checkpoint
                else:
                    print ("=> Validation Accuracy did not improve")
                #else:
                      
                    """if epoch == startepoch:
                        best_accuracy=valid_loss
                    else:
                        is_best = bool(acc < best_accuracy)#(acc.numpy() > best_accuracy.numpy())
                        best_accuracy = min(acc, best_accuracy)#torch.FloatTensor(min(acc.numpy(), best_accuracy.numpy()))
                        if is_best:
                            print ("=> Saving a new best")
                            save(epoch) # save checkpoint
                        else:
                            print ("=> Validation Accuracy did not improve")"""
            valid_losses.append(valid_loss)
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return
        
        
def train4(lr, model,train_loader, valid_loader,criterion, validation, init_optimizer, n_epochs=1, fold=1):
    optimizer = init_optimizer(lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5,patience=20)################
    #model = nn.DataParallel(model, device_ids=None)
    if torch.cuda.is_available():
        model.cuda()
    isrestore = False   
    startepoch = 0
    model_path = Path('model_{fold}.pt'.format(fold=fold))
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
        isrestore = True
        startepoch = epoch
        
        valid_metrics = validation(model, criterion, valid_loader)
        valid_loss = valid_metrics['valid_loss']
        ious_loss = valid_metrics['ious_loss']
        print(ious_loss)
        best_accuracy=-ious_loss
    else:
        epoch = 1
        step = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(model_path))
    

    report_each = 50
    log = open('train_{fold}.log'.format(fold=fold),'at', encoding='utf8')
    valid_losses = []
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        random.seed()
        tq = tqdm(total=len(train_loader) *  BATCH_SIZE)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        tl = train_loader
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                inputs, targets = variable(inputs), variable(targets)
                outputs,o2,o3,o4 = model(inputs)
                loss = criterion(outputs, o2, o3,o4, targets)
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss.data[0])
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                if i and i % report_each == 0:
                    write_event(log, lr, step, loss=mean_loss)
            write_event(log, lr, step, loss=mean_loss)########################lr
            tq.close()
            #save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader)
            write_event(log, lr, step, **valid_metrics)#############################
            valid_loss = valid_metrics['valid_loss']

            acc= -valid_metrics['ious_loss']
            #scheduler.step(acc)###########################
            #lr=get_learning_rate(optimizer)###############################
            
            if epoch==1:
                best_accuracy=-valid_metrics['ious_loss']#valid_loss#(valid_loss.from_numpy())
                is_best=True
                if is_best:
                    print ("=> Saving a new best")
                    save(epoch) # save checkpoint
                else:
                    print ("=> Validation Accuracy did not improve")
            else:
                #if isrestore == False:
                is_best = bool(acc < best_accuracy)#(acc.numpy() > best_accuracy.numpy())
                best_accuracy = min(acc, best_accuracy)#torch.FloatTensor(min(acc.numpy(), best_accuracy.numpy()))
                if is_best:
                    print ("=> Saving a new best")
                    save(epoch) # save checkpoint
                else:
                    print ("=> Validation Accuracy did not improve")
                #else:
                      
                    """if epoch == startepoch:
                        best_accuracy=valid_loss
                    else:
                        is_best = bool(acc < best_accuracy)#(acc.numpy() > best_accuracy.numpy())
                        best_accuracy = min(acc, best_accuracy)#torch.FloatTensor(min(acc.numpy(), best_accuracy.numpy()))
                        if is_best:
                            print ("=> Saving a new best")
                            save(epoch) # save checkpoint
                        else:
                            print ("=> Validation Accuracy did not improve")"""
            valid_losses.append(valid_loss)
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return
        
def train5(lr, model,train_loader, valid_loader,criterion, validation, init_optimizer, n_epochs=1, fold=1):
    optimizer = init_optimizer(lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5,patience=20)################
    #model = nn.DataParallel(model, device_ids=None)
    if torch.cuda.is_available():
        model.cuda()
    isrestore = False   
    startepoch = 0
    model_path = Path('model_{fold}.pt'.format(fold=fold))
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
        isrestore = True
        startepoch = epoch
        
        valid_metrics = validation(model, criterion, valid_loader)
        valid_loss = valid_metrics['valid_loss']
        ious_loss = valid_metrics['ious_loss']
        print(ious_loss)
        best_accuracy=-ious_loss
    else:
        epoch = 1
        step = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(model_path))
    

    report_each = 50
    log = open('train_{fold}.log'.format(fold=fold),'at', encoding='utf8')
    valid_losses = []
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        random.seed()
        tq = tqdm(total=len(train_loader) *  BATCH_SIZE)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        tl = train_loader
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                inputs, targets = variable(inputs), variable(targets)
                outputs,o2,o3,o4,o5 = model(inputs)
                loss = criterion(outputs, o2, o3, o4, o5, targets)
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss.data[0])
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                if i and i % report_each == 0:
                    write_event(log, lr, step, loss=mean_loss)
            write_event(log, lr, step, loss=mean_loss)########################lr
            tq.close()
            #save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader)
            write_event(log, lr, step, **valid_metrics)#############################
            valid_loss = valid_metrics['valid_loss']

            acc= -valid_metrics['ious_loss']
            #scheduler.step(acc)###########################
            #lr=get_learning_rate(optimizer)###############################
            
            if epoch==1:
                best_accuracy=-valid_metrics['ious_loss']#valid_loss#(valid_loss.from_numpy())
                is_best=True
                if is_best:
                    print ("=> Saving a new best")
                    save(epoch) # save checkpoint
                else:
                    print ("=> Validation Accuracy did not improve")
            else:
                #if isrestore == False:
                is_best = bool(acc < best_accuracy)#(acc.numpy() > best_accuracy.numpy())
                best_accuracy = min(acc, best_accuracy)#torch.FloatTensor(min(acc.numpy(), best_accuracy.numpy()))
                if is_best:
                    print ("=> Saving a new best")
                    save(epoch) # save checkpoint
                else:
                    print ("=> Validation Accuracy did not improve")
                #else:
                      
                    """if epoch == startepoch:
                        best_accuracy=valid_loss
                    else:
                        is_best = bool(acc < best_accuracy)#(acc.numpy() > best_accuracy.numpy())
                        best_accuracy = min(acc, best_accuracy)#torch.FloatTensor(min(acc.numpy(), best_accuracy.numpy()))
                        if is_best:
                            print ("=> Saving a new best")
                            save(epoch) # save checkpoint
                        else:
                            print ("=> Validation Accuracy did not improve")"""
            valid_losses.append(valid_loss)
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return


def retrain(lr, model, criterion, train_loader, valid_loader, validation, init_optimizer, n_epochs=1, fold=1):
    optimizer = init_optimizer(lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5,patience=20)################
    #model = nn.DataParallel(model, device_ids=None)
    if torch.cuda.is_available():
        model.cuda()
    isrestore = False   
    startepoch = 0
    model_path = Path('model_{fold}.pt'.format(fold=fold))
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
        isrestore = True
        startepoch = epoch
        
        valid_metrics = validation(model, criterion, valid_loader)
        valid_loss = valid_metrics['valid_loss']
        ious_loss = valid_metrics['ious_loss']
        print(ious_loss)
        best_accuracy=-ious_loss
    else:
        epoch = 2
        step = 0
        valid_metrics = validation(model, criterion, valid_loader)
        valid_loss = valid_metrics['valid_loss']
        #print(valid_loss)
        #best_accuracy=valid_loss
        ious_loss = valid_metrics['ious_loss']
        print(ious_loss)
        best_accuracy=-ious_loss        

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(model_path))

    print ("=> Saving a new best")###############################
    save(epoch) # save checkpoint########################

    report_each = 50
    log = open('train_{fold}.log'.format(fold=fold),'at', encoding='utf8')
    valid_losses = []
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        random.seed()
        tq = tqdm(total=len(train_loader) *  BATCH_SIZE)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        tl = train_loader
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                inputs, targets = variable(inputs), variable(targets)
                outputs = model(inputs)
                #outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss.data[0])
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                if i and i % report_each == 0:
                    write_event(log, lr, step, loss=mean_loss)######################
            write_event(log, lr, step, loss=mean_loss)########################
            tq.close()
            #save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader)
            write_event(log, lr, step, **valid_metrics)##############################
            valid_loss = valid_metrics['valid_loss']

            acc= -valid_metrics['ious_loss']
            scheduler.step(acc)###########################
            lr=get_learning_rate(optimizer)###############################
            
            if epoch==1:
                best_accuracy=-valid_metrics['ious_loss']#valid_loss#(valid_loss.from_numpy())
                is_best=True
                if is_best:
                    print ("=> Saving a new best")
                    save(epoch) # save checkpoint
                else:
                    print ("=> Validation Accuracy did not improve")
            else:
                #if isrestore == False:
                is_best = bool(acc < best_accuracy)#(acc.numpy() > best_accuracy.numpy())
                best_accuracy = min(acc, best_accuracy)#torch.FloatTensor(min(acc.numpy(), best_accuracy.numpy()))
                if is_best:
                    print ("=> Saving a new best")
                    save(epoch) # save checkpoint
                else:
                    print ("=> Validation Accuracy did not improve")
                #else:
                      
                    """if epoch == startepoch:
                        best_accuracy=valid_loss
                    else:
                        is_best = bool(acc < best_accuracy)#(acc.numpy() > best_accuracy.numpy())
                        best_accuracy = min(acc, best_accuracy)#torch.FloatTensor(min(acc.numpy(), best_accuracy.numpy()))
                        if is_best:
                            print ("=> Saving a new best")
                            save(epoch) # save checkpoint
                        else:
                            print ("=> Validation Accuracy did not improve")"""
            valid_losses.append(valid_loss)
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return

def retrain4(lr, model, train_loader, valid_loader,criterion, validation, init_optimizer, n_epochs=1, fold=1):
    optimizer = init_optimizer(lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5,patience=20)################
    #model = nn.DataParallel(model, device_ids=None)
    if torch.cuda.is_available():
        model.cuda()
    isrestore = False   
    startepoch = 0
    model_path = Path('model_{fold}.pt'.format(fold=fold))
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
        isrestore = True
        startepoch = epoch
        
        valid_metrics = validation(model, criterion, valid_loader)
        valid_loss = valid_metrics['valid_loss']
        ious_loss = valid_metrics['ious_loss']
        print(ious_loss)
        best_accuracy=-ious_loss
    else:
        epoch = 2
        step = 0
        valid_metrics = validation(model, criterion, valid_loader)
        valid_loss = valid_metrics['valid_loss']
        #print(valid_loss)
        #best_accuracy=valid_loss
        ious_loss = valid_metrics['ious_loss']
        print(ious_loss)
        best_accuracy=-ious_loss        

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(model_path))

    print ("=> Saving a new best")###############################
    save(epoch) # save checkpoint########################

    report_each = 50
    log = open('train_{fold}.log'.format(fold=fold),'at', encoding='utf8')
    valid_losses = []
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        random.seed()
        tq = tqdm(total=len(train_loader) *  BATCH_SIZE)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        tl = train_loader
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                inputs, targets = variable(inputs), variable(targets)
                outputs,o2,o3 ,o4= model(inputs)
                #outputs = model(inputs)
                loss = criterion(outputs, o2,o3,o4, targets)
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss.data[0])
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                if i and i % report_each == 0:
                    write_event(log, lr, step, loss=mean_loss)######################
            write_event(log, lr, step, loss=mean_loss)########################
            tq.close()
            #save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader)
            write_event(log, lr, step, **valid_metrics)##############################
            valid_loss = valid_metrics['valid_loss']

            acc= -valid_metrics['ious_loss']
            #scheduler.step(acc)###########################
            #lr=get_learning_rate(optimizer)###############################
            
            if epoch==1:
                best_accuracy=-valid_metrics['ious_loss']#valid_loss#(valid_loss.from_numpy())
                is_best=True
                if is_best:
                    print ("=> Saving a new best")
                    save(epoch) # save checkpoint
                else:
                    print ("=> Validation Accuracy did not improve")
            else:
                #if isrestore == False:
                is_best = bool(acc < best_accuracy)#(acc.numpy() > best_accuracy.numpy())
                best_accuracy = min(acc, best_accuracy)#torch.FloatTensor(min(acc.numpy(), best_accuracy.numpy()))
                if is_best:
                    print ("=> Saving a new best")
                    save(epoch) # save checkpoint
                else:
                    print ("=> Validation Accuracy did not improve")
                #else:
                      
                    """if epoch == startepoch:
                        best_accuracy=valid_loss
                    else:
                        is_best = bool(acc < best_accuracy)#(acc.numpy() > best_accuracy.numpy())
                        best_accuracy = min(acc, best_accuracy)#torch.FloatTensor(min(acc.numpy(), best_accuracy.numpy()))
                        if is_best:
                            print ("=> Saving a new best")
                            save(epoch) # save checkpoint
                        else:
                            print ("=> Validation Accuracy did not improve")"""
            valid_losses.append(valid_loss)
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return

def retrain5(lr, model, train_loader, valid_loader,criterion, validation, init_optimizer, n_epochs=1, fold=1):
    optimizer = init_optimizer(lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5,patience=20)################
    #model = nn.DataParallel(model, device_ids=None)
    if torch.cuda.is_available():
        model.cuda()
    isrestore = False   
    startepoch = 0
    model_path = Path('model_{fold}.pt'.format(fold=fold))
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
        isrestore = True
        startepoch = epoch
        
        valid_metrics = validation(model, criterion, valid_loader)
        valid_loss = valid_metrics['valid_loss']
        ious_loss = valid_metrics['ious_loss']
        print(ious_loss)
        best_accuracy=-ious_loss
    else:
        epoch = 2
        step = 0
        valid_metrics = validation(model, criterion, valid_loader)
        valid_loss = valid_metrics['valid_loss']
        #print(valid_loss)
        #best_accuracy=valid_loss
        ious_loss = valid_metrics['ious_loss']
        print(ious_loss)
        best_accuracy=-ious_loss        

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(model_path))

    print ("=> Saving a new best")###############################
    save(epoch) # save checkpoint########################

    report_each = 50
    log = open('train_{fold}.log'.format(fold=fold),'at', encoding='utf8')
    valid_losses = []
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        random.seed()
        tq = tqdm(total=len(train_loader) *  BATCH_SIZE)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        tl = train_loader
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                inputs, targets = variable(inputs), variable(targets)
                outputs,o2,o3 ,o4, o5= model(inputs)
                #outputs = model(inputs)
                loss = criterion(outputs, o2,o3,o4,o5, targets)
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss.data[0])
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                if i and i % report_each == 0:
                    write_event(log, lr, step, loss=mean_loss)######################
            write_event(log, lr, step, loss=mean_loss)########################
            tq.close()
            #save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader)
            write_event(log, lr, step, **valid_metrics)##############################
            valid_loss = valid_metrics['valid_loss']

            acc= -valid_metrics['ious_loss']
            #scheduler.step(acc)###########################
            #lr=get_learning_rate(optimizer)###############################
            
            if epoch==1:
                best_accuracy=-valid_metrics['ious_loss']#valid_loss#(valid_loss.from_numpy())
                is_best=True
                if is_best:
                    print ("=> Saving a new best")
                    save(epoch) # save checkpoint
                else:
                    print ("=> Validation Accuracy did not improve")
            else:
                #if isrestore == False:
                is_best = bool(acc < best_accuracy)#(acc.numpy() > best_accuracy.numpy())
                best_accuracy = min(acc, best_accuracy)#torch.FloatTensor(min(acc.numpy(), best_accuracy.numpy()))
                if is_best:
                    print ("=> Saving a new best")
                    save(epoch) # save checkpoint
                else:
                    print ("=> Validation Accuracy did not improve")
                #else:
                      
                    """if epoch == startepoch:
                        best_accuracy=valid_loss
                    else:
                        is_best = bool(acc < best_accuracy)#(acc.numpy() > best_accuracy.numpy())
                        best_accuracy = min(acc, best_accuracy)#torch.FloatTensor(min(acc.numpy(), best_accuracy.numpy()))
                        if is_best:
                            print ("=> Saving a new best")
                            save(epoch) # save checkpoint
                        else:
                            print ("=> Validation Accuracy did not improve")"""
            valid_losses.append(valid_loss)
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return

######################################################
def criterionL(output_4, output_3, output_2, output_1, truth_pixel, is_average=True):
    #print(torch.sum(torch.sum(truth_pixel)))
    bat,dim,m,n=truth_pixel.shape


    loss_logit1 = lovasz_loss(output_1, truth_pixel)#1-1-1
    loss_logit2 = lovasz_loss(output_2, truth_pixel)#1-1-1
    loss_logit3 = lovasz_loss(output_3, truth_pixel)#1-1-1
    loss_logit4 = lovasz_loss(output_4, truth_pixel)#1-1-1

    #weight_image, weight_pixel = 0.1, 10  #focal
    weight_1, weight_2, weight_3, weight_4 = 0.2, 0.2, 0.2, 1  #lovasz?
    #weight_image, weight_pixel = 0.1, 2 #bce



    return weight_1*loss_logit1+ weight_2*loss_logit2+weight_3*loss_logit3+weight_4*loss_logit4

def criterionL5(output_4, output_3, output_2, output_1, logit_image, truth_pixel, is_average=True):
    #print(torch.sum(torch.sum(truth_pixel)))
    bat,dim,m,n=truth_pixel.shape

    truth_image = torch.Tensor(bat)
    i=0
    for x in truth_pixel:
        truth_image[i]=(x.squeeze().sum()>0).cuda().float()
        i=i+1

    truth_image=truth_image.type(torch.cuda.FloatTensor)

    loss_image = F.binary_cross_entropy_with_logits(logit_image, truth_image, reduce=is_average)#1-1-1
    
    loss_logit1 = lovasz_loss(output_1, truth_pixel)#1-1-1
    loss_logit2 = lovasz_loss(output_2, truth_pixel)#1-1-1
    loss_logit3 = lovasz_loss(output_3, truth_pixel)#1-1-1
    loss_logit4 = lovasz_loss(output_4, truth_pixel)#1-1-1

    #weight_image, weight_pixel = 0.1, 10  #focal
    weight_1, weight_2, weight_3, weight_4, weight_image = 0.1, 0.1, 0.1, 1, 0.05  #0.2, 0.2, 0.2, 1, 0.05  #lovasz?
    #weight_image, weight_pixel = 0.1, 2 #bce



    return weight_1*loss_logit1+ weight_2*loss_logit2+weight_3*loss_logit3+weight_4*loss_logit4+weight_image*loss_image
########################################################


