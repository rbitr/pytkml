import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
import numpy as np

from torch.nn.utils import parameters_to_vector as params2vec
from collections import defaultdict
from loguru import logger

from pathlib import Path
import json

from typing import Iterable

from pytkml.misc.logging import arrayToBase64IM

LABELS = lambda x: x[1]
SAMPLES = lambda x: x[0]
IDENTITY = lambda x: x

# Reducers for converting output probabilities to a single value
def first_element_reducer(x):
    return torch.Tensor([x[0] for x in x])

def first_element_reducerInt(x):
    return torch.IntTensor([x[0] for x in x])

def argmax_reducer(x):
    return torch.argmax(x,dim=1)

# tess for equality
def element_wise_equal(sample_out,label_out):
    return torch.all(torch.eq(sample_out,label_out))

def accuracy_threshold(thresh):
    def compare(sample_out,label_out):
        return accuracy(sample_out,label_out) > thresh
    return compare


def calc_gradients(loss_func,params,Loader):
    #model.eval()

    gradients = []
    train_labels = []
    train_samples = []

    tl = Loader()
    for t_samp, t_lab in tl:
        for s, l in zip(t_samp, t_lab):
            train_labels.append(l)
            train_samples.append(s)
            loss = loss_func((s.unsqueeze(0),l.unsqueeze(0)))
            grad_train = params2vec(torch.autograd.grad(loss,params,retain_graph=False))
            #train_g = grad_train / torch.linalg.norm(grad_train)
            gradients.append(grad_train)
            #I_c.append(np.dot(-test_g.detach(),train_g.detach()))

    return gradients, train_samples, train_labels

def gradientCosine(grad_test, grad_train):
    test_g = grad_test / torch.linalg.norm(grad_test)
    train_g = grad_train / torch.linalg.norm(grad_train)
    return np.dot(-test_g.detach(),train_g.detach())

def influence_transform(model,trainLoader,slice=0,criterion=None,verbose=True): # it actually probably needs the ModelTester object, because it needs the loaders

    def transform(batch):
        model.eval()
        # calculate the gradients for each training point first
        if hasattr(model,'test_step'):
            loss_func = lambda x: model.test_step(x,-1) # requires a batch and an ID
            logger.info("Using loss from model test step")
        else:
            def loss_func(batch):
                samples, labels = batch
                outputs = model(samples)
                loss = criterion(outputs, labels)
                return loss
            logger.info("Using user supplied loss criteria")
            #raise NotImplementedError

        params = [p for name,p in model.named_parameters()]

        train_gradients, train_samples, train_labels = calc_gradients(loss_func,params,trainLoader)

        test_gradients, test_samples, test_labels = calc_gradients(loss_func,params,lambda :[batch])

        closest_points = []


        for grad_test, samp, lab in zip(test_gradients, test_samples, test_labels):

            #I_c = []
            #logger.info(f"sample shape is {samp.shape}")
            pred = model(samp.unsqueeze(0))
            test_g = grad_test / torch.linalg.norm(grad_test)

            I_c = [gradientCosine(grad_test, grad_train) for grad_train in train_gradients]
            #for grad_train, s, l in zip(train_gradients, train_samples, train_labels):

            #    I_c.append(gradientCosin(grad_test, grad_train))

            # log the label, the prediction, the confidence, and the influential image
            pred_np = pred.squeeze().detach().numpy()
            label_prob = pred_np[lab] # what dimension is this?
            #logger.info(str(label_prob))
            inf_ids = np.argsort(I_c)[slice]
            if not isinstance(inf_ids, Iterable):
                #print(f"inf_ids is {inf_ids}, type is {type(inf_ids)}")
                inf_ids = [inf_ids]

            inf_labels = [int(train_labels[inf_id]) for inf_id in inf_ids]
            inf_samples = [arrayToBase64IM(train_samples[inf_id].numpy()) for inf_id in inf_ids]
            sample_dict = {"true_label":int(lab),
                           "pred_label":int(np.argmax(pred_np)),
                           "sample":arrayToBase64IM(samp),
                           "confidence":str(label_prob),
                           "closest_label":inf_labels,
                           "closest_sample":inf_samples}

            #for r in sample_dict.keys():
            #    logger.info(r+ " " + str(type(sample_dict[r])))

            verbose and logger.info(json.dumps(sample_dict))
            #logger.info("Some INFO!!")

            closest_points.append(inf_labels)

        return torch.IntTensor(closest_points)

    return transform
