import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, RandomSampler
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
import numpy as np



def RandomSubsetLoader(dataset, num, batch_size=16,seed=0):

    def get_loader():
        subset, _ = random_split(dataset,[num, len(dataset)-num],generator=torch.Generator().manual_seed(seed))
        return DataLoader(subset,batch_size=batch_size,shuffle=False)

    return get_loader
