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
from pytkml.core.processors import IDENTITY, SAMPLES, LABELS
from pytkml.misc.logging import arrayToBase64IM

#class TestArgs:
#
#    def __init__(self,name=""):
#        self.name = name
#
#    def __getattr__(self, item):
#            return None
def test_args_dict(**init_dict):
    return defaultdict(lambda: None,**init_dict)

class ModelTester():

    def __init__(self,model=None,test_dataloader=None,train_dataloader=None,val_dataloader=None,logdir="./logs"):
        self.model = model
        self.test_dataloader = test_dataloader or model.test_dataloader
        self.train_dataloader = train_dataloader or model.train_dataloader
        self.val_dataloader = val_dataloader or model.val_dataloader

        self.tests = []

        logger.add(Path(logdir) / "logfile_{time}.log")

    def sample_pass(self,batch,transform,alternateForward=None,**kwargs):
        batch_sample, batch_label = batch
        self.model.eval()
        forward = alternateForward or (lambda b: self.model(b).detach())
        return transform((forward(batch_sample),batch_label),**kwargs)

    def label_pass(self,batch,transform,**kwargs):
        return transform(batch, **kwargs)

    def test_loop(self,batch,test_dict,**kwargs):


        sample_reduce = test_dict['sample_reduce'] or IDENTITY
        sample_xform = test_dict['sample_transform'] or SAMPLES
        label_reduce = test_dict['label_reduce'] or IDENTITY
        label_xform = test_dict['label_transform'] or LABELS
        comparison = test_dict['comparison'] or (lambda x,y: x==y)


        sample_out = sample_reduce(self.sample_pass(batch,sample_xform,alternateForward=test_dict['forward'],**kwargs))

        label_out = label_reduce(self.label_pass(batch,label_xform, **kwargs))

        logger.info("Test: " + test_dict['name'] or "unnamed")
        logger.info("Sample Output: " + str(sample_out))
        logger.info("Label Output: " + str(label_out))
        assert comparison(sample_out,label_out)
        logger.info("Test Passed")

    def test(self):
        batch = next(iter(self.test_dataloader()))

        for t in self.tests:
            self.test_loop(batch,t)
