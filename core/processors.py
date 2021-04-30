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

from pytkml.misc.logging import arrayToBase64IM

LABELS = lambda x: x[1]
SAMPLES = lambda x: x[0]
IDENTITY = lambda x: x

# Reducers for converting output probabilities to a single value
def first_element_reducer(x):
    return torch.Tensor([x[0] for x in x])

def argmax_reducer(x):
    return torch.argmax(x,dim=1)

# tess for equality
def element_wise_equal(sample_out,label_out):
    return torch.all(torch.eq(sample_out,label_out))

def accuracy_threshold(thresh):
    def compare(sample_out,label_out):
        return accuracy(sample_out,label_out) > thresh
    return compare


def calc_gradients(model,Loader):
    model.eval()
    params = [p for name,p in model.named_parameters()]
    gradients = []
    train_labels = []
    train_samples = []

    tl = Loader()
    for t_samp, t_lab in tl:
        for s, l in zip(t_samp, t_lab):
            train_labels.append(l)
            train_samples.append(s)
            loss = model.validation_step((s.unsqueeze(0),l.unsqueeze(0)),-1)
            grad_train = params2vec(torch.autograd.grad(loss,params,retain_graph=False))
            #train_g = grad_train / torch.linalg.norm(grad_train)
            gradients.append(grad_train)
            #I_c.append(np.dot(-test_g.detach(),train_g.detach()))

    return gradients, train_samples, train_labels

def gradientCosine(grad_test, grad_train):
    test_g = grad_test / torch.linalg.norm(grad_test)
    train_g = grad_train / torch.linalg.norm(grad_train)
    return np.dot(-test_g.detach(),train_g.detach())

def influence_transform(model,trainLoader,slice=0,verbose=True): # it actually probably needs the ModelTester object, because it needs the loaders

    def transform(batch):

        # calculate the gradients for each training point first
        train_gradients, train_samples, train_labels = calc_gradients(model,trainLoader)

        test_gradients, test_samples, test_labels = calc_gradients(model,lambda :[batch])

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

            label_prob = pred.squeeze().detach().numpy()[lab] # what dimension is this?
            #logger.info(str(label_prob))
            inf_id = np.argsort(I_c)[slice]
            inf_label = int(train_labels[inf_id])
            inf_sample = train_samples[inf_id].numpy()
            sample_dict = {"true_label":int(lab),
                           "sample":arrayToBase64IM(samp),
                           "confidence":str(label_prob),
                           "closest_label":inf_label,
                           "closest_sample":arrayToBase64IM(inf_sample)}

            #for r in sample_dict.keys():
            #    logger.info(r+ " " + str(type(sample_dict[r])))

            verbose and logger.info(json.dumps(sample_dict))
            #logger.info("Some INFO!!")

            closest_points.append(inf_label)

        return torch.IntTensor(closest_points)

    return transform


def influence_transform_old(model,verbose=True): # it actually probably needs the ModelTester object, because it needs the loaders
    def transform(batch):

        closest_points = []
        for samp, lab in zip(*batch):

            pred = model(samp.unsqueeze(0))
            # will only work for LightningModule
            loss = model.training_step((samp.unsqueeze(0),lab.unsqueeze(0)),-1).cpu() # view?
            params = [p for name,p in model.named_parameters()]
            grad_test = params2vec(torch.autograd.grad(loss,params, create_graph=True, retain_graph=True))

            test_g = grad_test / torch.linalg.norm(grad_test)

            I_c = []
            train_labels = []
            train_samples = []
            tl = model.train_dataloader() # Should be some special subset of train set
            tl_iter = iter(tl)

            N_ITERS = 2 # needs to change

            for _ in range(N_ITERS):

                t_samp, t_lab = next(tl_iter)
                for s, l in zip(t_samp, t_lab):
                    train_labels.append(l)
                    train_samples.append(s)
                    loss = model.validation_step((s.unsqueeze(0),l.unsqueeze(0)),-1)
                    grad_train = params2vec(torch.autograd.grad(loss,params,retain_graph=False))
                    train_g = grad_train / torch.linalg.norm(grad_train)
                    I_c.append(np.dot(-test_g.detach(),train_g.detach()))

            # log the label, the prediction, the confidence, and the influential image

            label_prob = pred.squeeze().detach().numpy()[lab] # what dimension is this?
            #logger.info(str(label_prob))
            inf_id = np.argsort(I_c)[0]
            inf_label = int(train_labels[inf_id])
            inf_sample = train_samples[inf_id].numpy()
            sample_dict = {"true_label":int(lab),
                           "sample":arrayToBase64IM(samp),
                           "confidence":str(label_prob),
                           "closest_label":inf_label,
                           "closest_sample":arrayToBase64IM(inf_sample)}

            #for r in sample_dict.keys():
            #    logger.info(r+ " " + str(type(sample_dict[r])))

            verbose and logger.info(json.dumps(sample_dict))
            #logger.info("Some INFO!!")

            closest_points.append(inf_label)


        return torch.IntTensor(closest_points)

    return transform
