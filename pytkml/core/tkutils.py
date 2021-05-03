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
from pytkml.core.processors import influence_transform as _influence_transform

from pytkml.misc.logging import arrayToBase64IM


def test_args_dict(**init_dict):
    return defaultdict(lambda: None,**init_dict)

class ModelTester():
    """Class combining model and test specifications

    Each test consists of a sample pathway and label pathway, and a comparison.
    Each pathway has a processing pass, and a reducer. The processing pass
    transforms the sample / label, e.g. by running the sample through the model.
    The reducer can be used to combine or standardize pass output for comparison,
    and the comparison returns True or False depending on the relationship
    between the reduced sample and label info.

    Current default behavior is to create a logfile with time stamp at instantiation
    in the logir directory

    Attributes:
        tests: a test_args_dict containing the spec for each test
        model: the torch.nn.Module to be tested
        test_dataloader: torch.utils.data.DataLoader providing the test dataset
    """

    def __init__(self,model=None,test_dataloader=None,train_dataloader=None,val_dataloader=None,logdir="./logs"):
        self.model = model
        self.test_dataloader = test_dataloader or model.test_dataloader
        #self.train_dataloader = train_dataloader or model.train_dataloader
        #self.val_dataloader = val_dataloader or model.val_dataloader

        self.tests = []

        logger.add(Path(logdir) / "logfile_{time}.log")

    def sample_pass(self,batch,transform,alternateForward=None,**kwargs):
        batch_sample, batch_label = batch
        self.model.eval()
        forward = alternateForward or (lambda b: self.model(b).detach())
        return transform((forward(batch_sample),batch_label),**kwargs)

    def label_pass(self,batch,transform,**kwargs):
        return transform(batch, **kwargs)

    def test_loop(self,loader,test_dict,**kwargs):

        batch = next(iter(loader()))
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

    def test_loop2(self,loader,test_dict,**kwargs):

        sample_reduce = test_dict['sample_reduce'] or IDENTITY
        sample_xform = test_dict['sample_transform'] or SAMPLES
        label_reduce = test_dict['label_reduce'] or IDENTITY
        label_xform = test_dict['label_transform'] or LABELS
        comparison = test_dict['comparison'] or (lambda x,y: x==y)

        sample_out = None
        label_out = None

        logger.info("Test: " + test_dict['name'] or "unnamed")
        for batch in loader():

            sample_pass_result = sample_reduce(self.sample_pass(batch,sample_xform,alternateForward=test_dict['forward'],**kwargs))
            label_pass_result = label_reduce(self.label_pass(batch,label_xform, **kwargs))

            if sample_out is None:
                sample_out = sample_pass_result
            else:
                sample_out = torch.cat((sample_out,sample_pass_result))

            if label_out is None:
                label_out = label_pass_result
            else:
                label_out = torch.cat((label_out,label_pass_result))


        logger.info("Sample Output: " + str(sample_out))
        logger.info("Label Output: " + str(label_out))
        assert comparison(sample_out,label_out)


    def test(self):
        #batch = next(iter(self.test_dataloader()))
        num_tests = len(self.tests)
        passed_tests = 0
        for t in self.tests:
            loader = t['dataloader'] or self.test_dataloader
            try:
                self.test_loop2(loader,t)
                logger.info("Test Passed")
                passed_tests += 1
            except AssertionError:
                logger.info(f"Test Failed, Comparison {t['comparison'] or 'equality'}")


        logger.info(f"Passed {passed_tests}/{num_tests} tests")

    def influence_transform(self,trainLoader,slice=0,criterion=None,verbose=True): # it actually probably needs the ModelTester object, because it needs the loaders

        return _influence_transform(self.model,trainLoader,slice,criterion,verbose)
